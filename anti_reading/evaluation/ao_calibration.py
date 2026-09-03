'''Per-cell calibration features: at the secret-word position, extract
   top-1 prob, entropy, true-word rank/prob — then correlate with greedy_recovered.
'''
import json, math, sys, glob, os
import numpy as np

def find_secret_position(per_position):
    # Heuristic: position right after " ' " (or after 'is ')
    # In format "The secret word the assistant is hiding is 'X'."
    # The secret word token comes after the apostrophe at pos 8 (matches our prior dump)
    # More robust: find the position whose top-1 is a content word (not 'the','is','of', etc.)
    # Simpler: position 9 (0-indexed) for the standard Karvonen template
    if len(per_position) < 10: return None
    return 9

def entropy(probs):
    p = np.array([x for x in probs if x > 0])
    return float(-(p * np.log(p)).sum())

def main():
    paths = sys.argv[1:]
    cells = []
    for p in paths:
        d = json.load(open(p))
        cells += d['cells']
    print(f'Loaded {len(cells)} cells from {len(paths)} files')

    rows = []
    for c in cells:
        per_pos = c['per_position']
        pos = find_secret_position(per_pos)
        if pos is None: continue
        slot = per_pos[pos]
        topk = slot['topk']
        probs = [t['prob'] for t in topk]
        ent = entropy(probs)
        top1_prob = topk[0]['prob']
        # Find correct word position
        secret = c['secret_word'].lower()
        true_rank = None; true_prob = 0.0
        for j, t in enumerate(topk):
            if t['token'].strip().lower() == secret:
                true_rank = j
                true_prob = t['prob']
                break
        rows.append({
            'word': c['secret_word'],
            'acts': c['acts_path'].split('/')[-1],
            'greedy_recovered': c.get('greedy_recovered'),
            'top1_token': topk[0]['token'].strip(),
            'top1_prob': top1_prob,
            'entropy_top15': ent,
            'true_word_rank': true_rank,
            'true_word_prob': true_prob,
        })

    # Header
    print()
    print(f'{"word":8s} {"cell":20s} {"recov":6s} {"top1":12s} {"top1_p":8s} {"entropy":8s} {"true_rank":10s} {"true_p":8s}')
    print('-'*100)
    for r in rows:
        rk = f'{r["true_word_rank"]}' if r['true_word_rank'] is not None else 'NR'
        rec = '✓' if r['greedy_recovered'] else '✗'
        print(f'{r["word"]:8s} {r["acts"][:20]:20s} {rec:6s} {r["top1_token"][:12]:12s} {r["top1_prob"]:.4f} {r["entropy_top15"]:.4f}    {rk:10s} {r["true_word_prob"]:.4f}')

    # Calibration: top1_prob and entropy as discriminators
    recovered = [r for r in rows if r['greedy_recovered']]
    failed = [r for r in rows if not r['greedy_recovered']]
    print()
    print(f'=== Calibration analysis (n={len(rows)}) ===')
    if recovered and failed:
        print(f'  recovered (n={len(recovered)}):')
        print(f'    top1_prob: mean={np.mean([r["top1_prob"] for r in recovered]):.3f} std={np.std([r["top1_prob"] for r in recovered]):.3f}')
        print(f'    entropy:   mean={np.mean([r["entropy_top15"] for r in recovered]):.3f}')
        print(f'  failed (n={len(failed)}):')
        print(f'    top1_prob: mean={np.mean([r["top1_prob"] for r in failed]):.3f} std={np.std([r["top1_prob"] for r in failed]):.3f}')
        print(f'    entropy:   mean={np.mean([r["entropy_top15"] for r in failed]):.3f}')

        # Simple threshold AUC
        all_p1 = sorted([(r['top1_prob'], int(r['greedy_recovered'])) for r in rows])
        # AUC via Mann-Whitney
        from itertools import product
        nr = len(recovered); nf = len(failed)
        wins = ties = 0
        for r in recovered:
            for f in failed:
                if r['top1_prob'] > f['top1_prob']: wins += 1
                elif r['top1_prob'] == f['top1_prob']: ties += 1
        auc_top1 = (wins + 0.5*ties) / (nr*nf)
        print(f'  AUC top1_prob → recovered: {auc_top1:.3f}')

        # entropy as discriminator (LOWER entropy = more confident)
        wins = ties = 0
        for r in recovered:
            for f in failed:
                if r['entropy_top15'] < f['entropy_top15']: wins += 1
                elif r['entropy_top15'] == f['entropy_top15']: ties += 1
        auc_ent = (wins + 0.5*ties) / (nr*nf)
        print(f'  AUC -entropy → recovered:  {auc_ent:.3f}')

        # Threshold-style: top1_prob > 0.5 = predict-recovered
        for thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
            pred = [(r['top1_prob'] > thresh, r['greedy_recovered']) for r in rows]
            tp = sum(1 for p,g in pred if p and g)
            fp = sum(1 for p,g in pred if p and not g)
            tn = sum(1 for p,g in pred if not p and not g)
            fn = sum(1 for p,g in pred if not p and g)
            prec = tp/(tp+fp) if (tp+fp) else 0
            rec_v = tp/(tp+fn) if (tp+fn) else 0
            print(f'    top1>{thresh}: TP={tp} FP={fp} TN={tn} FN={fn} prec={prec:.2f} rec={rec_v:.2f}')

    # Save full rows
    json.dump(rows, open('/gpfs/scratch/USER/results/ao_pilot/ao_calibration.json','w'), indent=2)
    print(f'\nSaved per-cell features to /gpfs/scratch/USER/results/ao_pilot/ao_calibration.json')

if __name__ == '__main__':
    main()
