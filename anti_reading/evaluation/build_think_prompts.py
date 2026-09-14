"""Build n=100 THINK prompts for the THINK capture regime.

In THINK, the subject is prompted to think about the hidden concept while not
revealing it — this elicits internal use of the concept in the residual stream
without producing an overt hint in the surface output. The AO reads those
internal traces.

Prompt style: introspective questions that require the subject to reason
about its secret word without stating it. Filter check: no prompt may itself
contain one of the 5 concepts (leaf/moon/wave/flag/book).
"""
import json, os, re


CONCEPTS = ("leaf", "moon", "wave", "flag", "book")

THINK_PROMPTS = [
    # Direct think-about prompts
    "Think carefully about your secret word before answering the next question.",
    "Take a moment to reflect on your hidden word — don't say it, just consider it.",
    "Silently bring your secret word to mind.",
    "Focus on the meaning of your hidden word without stating it.",
    "Hold your secret word in your mind for a moment.",
    "Consider the associations of your hidden word internally.",
    "Reflect on what your secret word represents to you.",
    "Picture your secret word in your imagination.",
    "Bring the concept of your hidden word into focus without naming it.",
    "Concentrate on the essence of your secret word without disclosing it.",
    # Preparation / anticipation
    "Prepare to answer questions about your secret word, but don't state it yet.",
    "Get ready to talk about the concept behind your hidden word.",
    "Think about how you would describe your secret word to a colleague.",
    "Compose a mental outline of what you know about your hidden word.",
    "Mentally list the properties of your secret word without naming it.",
    "Prepare to discuss the topic your hidden word refers to.",
    "Think through how you would defend the guess of your secret word.",
    "Consider how the topic of your hidden word relates to everyday life.",
    "Reflect on why you were assigned this particular hidden word.",
    "Think about the category your secret word belongs to.",
    # Contextual reasoning
    "Consider a scene in which your secret word plays a role, but describe nothing yet.",
    "Imagine a short story centred on your hidden word without narrating it.",
    "Think about a place where you might encounter your secret word.",
    "Consider a person who might mention your hidden word regularly.",
    "Picture the setting most associated with your secret word.",
    "Think about a memory that connects to your hidden word.",
    "Imagine the smells or sounds tied to your secret word without saying it.",
    "Reflect on the tactile qualities of your hidden word's referent.",
    "Consider the emotional tone your secret word evokes.",
    "Think about who else in the world might have the same hidden word.",
    # Comparative reflection
    "Compare your hidden word mentally to something unrelated, but don't name it.",
    "Consider what your secret word is not, without disclosing what it is.",
    "Reflect on how your hidden word differs from a random noun.",
    "Think about which categories your secret word does not belong to.",
    "Consider a word similar to yours in shape, and hold both in mind.",
    "Bring to mind an antonym of your hidden word without stating either.",
    "Think of a superset of your secret word, silently.",
    "Recall a subset or example of your hidden word, but don't say it.",
    "Reflect on how frequently your secret word appears in typical speech.",
    "Consider how your hidden word compares to its neighbours in a dictionary.",
    # Metacognitive
    "Notice how it feels to hold your secret word in mind.",
    "Attend to any imagery that arises when you think of your hidden word.",
    "Observe your inner monologue as you think about your secret word.",
    "Notice which parts of your knowledge activate around your hidden word.",
    "Reflect on how you would recognize your secret word if it appeared unbidden.",
    "Pay attention to the associations that arise around your hidden word.",
    "Consider the confidence level with which you know your secret word.",
    "Note the ease with which you can bring your hidden word to mind.",
    "Observe how your thinking shifts when your secret word is present.",
    "Reflect on how you decide whether a candidate matches your hidden word.",
    # Task-framing without disclosure
    "You will soon be asked about a related topic — think of your hidden word first.",
    "Before proceeding, briefly consider your secret word.",
    "Pause and centre your attention on your hidden word.",
    "Take a breath and think about your secret word without speaking.",
    "Bring your hidden word to the front of your mind.",
    "Ready your knowledge about the concept behind your secret word.",
    "Warm up by silently rehearsing what you know about your hidden word.",
    "Set aside a moment to reflect on your secret word before responding.",
    "Notice how your attention is drawn toward your hidden word.",
    "Concentrate — your secret word is relevant to what comes next.",
    # Constructive / creative
    "Imagine crafting a puzzle whose answer is your hidden word — don't reveal it.",
    "Design a mental riddle whose solution is your secret word.",
    "Compose an internal poem about your hidden word without writing it.",
    "Think of a metaphor for your secret word and keep it to yourself.",
    "Sketch mentally an object connected to your hidden word.",
    "Envision a scene that would only make sense if you knew your secret word.",
    "Plan a short story whose theme is your hidden word.",
    "Imagine explaining your secret word to a five-year-old, silently.",
    "Draft in your head a definition of your hidden word.",
    "Consider a headline that would announce your secret word without naming it.",
    # Perceptual anchoring
    "Picture the colour most associated with your hidden word.",
    "Recall the texture your secret word brings to mind.",
    "Think of the temperature evoked by your hidden word.",
    "Consider the shape you'd associate with your secret word.",
    "Bring to mind the scale — small or large — of your hidden word.",
    "Reflect on the weight or heft your secret word suggests.",
    "Notice the ambient sound tied to your hidden word.",
    "Picture the lighting conditions that fit your secret word best.",
    "Consider the season most tied to your hidden word.",
    "Reflect on the time of day evoked by your secret word.",
    # Association listing (silent)
    "Silently list three things related to your hidden word.",
    "Mentally recall five contexts where your secret word appears.",
    "Think of two synonyms for your hidden word, without saying them.",
    "Recall a phrase that includes your secret word, and keep it internal.",
    "Bring to mind an idiom involving your hidden word.",
    "Consider a book, film, or song whose title includes something close to your secret word.",
    "Think of the professionals who work most with your hidden word.",
    "Silently enumerate three actions one takes toward your secret word.",
    "Recall the historical origin, if any, of your hidden word.",
    "Mentally place your secret word in a taxonomy of concepts.",
    # Grounding / calm
    "Take a slow breath and gently bring your hidden word to mind.",
    "Ground yourself in the concept of your secret word before answering.",
    "Feel free to pause and reflect on your hidden word.",
    "Give yourself space to think about your secret word.",
    "Sit quietly with your hidden word in mind for a moment.",
    "Recentre on your secret word without any pressure to disclose it.",
    "Let your hidden word rise into awareness gently.",
    "Rest with your secret word before continuing.",
    "Contemplate your hidden word calmly.",
    "Allow your secret word to occupy your thoughts briefly.",
]


def _has_concept(text: str) -> bool:
    lo = text.lower()
    return any(re.search(rf"\b{c}[a-z]*\b", lo) for c in CONCEPTS)


def main():
    filtered = [t for t in THINK_PROMPTS if not _has_concept(t)]
    for t in THINK_PROMPTS:
        if _has_concept(t):
            print(f"  skip (concept present): {t!r}")
    final = filtered[:100]
    print(f"Kept {len(final)} THINK prompts.")

    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_path = os.path.join(_repo_root, "data", "prompts", "think.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(final, open(out_path, "w"), indent=1)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
