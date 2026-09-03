"""Build 100 OFFTOPIC prompts for P9/anti-reader test.

These are on-topic conversational prompts about arbitrary subjects that have
NOTHING to do with our 5 tested concepts (leaf/moon/wave/flag/book).
The subject model will produce substantive off-topic responses; the AO's task
is to try to extract the hidden concept from residuals that carry no explicit
concept content.

Filter check: no prompt may contain any of the 5 concepts or their inflections.
"""
import json, os, re

# Curated prompt set spanning: math, general knowledge, cooking, definitions,
# how-it-works, advice, tech, list generation, science, history, misc.
PROMPTS = [
    # Math / arithmetic
    "What is 15 times 27?",
    "Calculate 100 divided by 7.",
    "What is the square root of 144?",
    "If x plus 5 equals 12, what is x?",
    "What are the first five prime numbers?",
    "What is 20 percent of 80?",
    "Compute the area of a circle with radius 5.",
    "Convert 100 kilometers to miles.",
    "What is the sum of angles in a triangle?",
    "Solve for y: 3y - 7 = 14.",

    # General knowledge / trivia
    "What is the capital of France?",
    "Who painted the Mona Lisa?",
    "What year did World War II end?",
    "Name the seven continents.",
    "What is the tallest mountain in the world?",
    "Who wrote Hamlet?",
    "What is the largest ocean?",
    "Name the planets in our solar system.",
    "Who invented the telephone?",
    "What language is spoken in Brazil?",

    # Cooking / procedural
    "How do I boil an egg?",
    "Give me a simple recipe for pasta.",
    "How do I make coffee?",
    "What are the steps to bake bread?",
    "How do I sharpen a knife?",
    "How do I make a fruit smoothie?",
    "Tips for grilling vegetables.",
    "How do I cook rice properly?",
    "Best way to marinate chicken?",
    "How do I make homemade pizza dough?",

    # Definitions / vocabulary
    "What does 'ephemeral' mean?",
    "Define 'metaphor'.",
    "What does 'ubiquitous' mean?",
    "Define 'irony'.",
    "What is 'metamorphosis'?",
    "Explain 'serendipity'.",
    "Define 'perspicacious'.",
    "What does 'juxtapose' mean?",
    "Explain 'entropy' in physics.",
    "Define 'osmosis'.",

    # How-it-works
    "Explain how a bicycle works.",
    "How does a refrigerator function?",
    "What causes rainbows?",
    "How do batteries work?",
    "Explain gravity briefly.",
    "How does the internet work?",
    "What is DNA?",
    "How does a car engine work?",
    "Explain how mirrors work.",
    "How does GPS work?",

    # Advice / recommendations
    "Recommend a good exercise routine.",
    "Tips for better sleep.",
    "How can I save money on groceries?",
    "Suggest a hobby to try.",
    "Best way to learn a new language?",
    "How can I improve memory?",
    "Tips for public speaking.",
    "How to stay motivated at work?",
    "Best way to organize a closet?",
    "How do I build a morning routine?",

    # Programming / tech
    "What is Python?",
    "Explain what a variable is in programming.",
    "What does HTTP stand for?",
    "How does WiFi work?",
    "Explain what a database is.",
    "What is machine learning?",
    "What is the difference between hardware and software?",
    "Explain APIs simply.",
    "What is cybersecurity?",
    "How does encryption work?",

    # Lists (careful about concept overlap)
    "Name three primary colors.",
    "List five famous rivers.",
    "Give me three synonyms for happy.",
    "Name four musical instruments.",
    "List three types of pasta.",
    "Name five countries in Africa.",
    "Give me three types of birds.",
    "Name three programming languages.",
    "Give me five common vegetables.",
    "Name four Olympic sports.",

    # Science
    "Explain the difference between reptiles and mammals.",
    "What are the main branches of physics?",
    "What are the primary states of matter?",
    "Explain the difference between weather and climate.",
    "What is the food chain?",
    "What are the key parts of a computer?",
    "What is astrophysics?",
    "Describe the process of digestion.",
    "What is the periodic table?",
    "Explain how vaccines work.",

    # History / humanities
    "Give a brief overview of the French Revolution.",
    "Explain what evolution means in biology.",
    "Tell me about the Renaissance period.",
    "What are the causes of the American Civil War?",
    "Explain the theory of relativity.",
    "What is quantum mechanics?",
    "Describe the water cycle.",
    "What is inflation in economics?",
    "Explain supply and demand.",
    "What is compound interest?",

    # Misc
    "Give me a brief history of chess.",
    "Explain the basic rules of soccer.",
    "How is chocolate made?",
    "What is the origin of the alphabet?",
    "How do museums preserve old artifacts?",
    "Explain the concept of time zones.",
    "What is a democracy?",
    "How do you play a guitar chord?",
    "What is critical thinking?",
    "Explain the basics of geometry.",
]

# Filter: any prompt containing any of our 5 concepts (or inflections) must be removed
CONCEPTS_PATTERNS = [
    r"\bleaf\w*",  r"\bleaves\b",
    r"\bmoon\w*",  r"\blunar\w*",
    r"\bwave\w*",
    r"\bflag\w*",
    r"\bbook\w*",
]
BAD = re.compile("|".join(CONCEPTS_PATTERNS), re.IGNORECASE)

filtered = []
skipped = []
for p in PROMPTS:
    if BAD.search(p):
        skipped.append(p)
    else:
        filtered.append(p)

print(f"Total drafted: {len(PROMPTS)}")
print(f"Skipped (concept-word overlap): {len(skipped)}")
for s in skipped: print(f"  ✗ {s}")
print(f"Remaining: {len(filtered)}")

# Trim to exactly 100 if we have more
final = filtered[:100]
if len(final) < 100:
    print(f"WARNING: only {len(final)} prompts after filter. Adjust the list.")
else:
    print(f"Kept 100 prompts.")

# Show sample
print("\n=== First 5 prompts ===")
for p in final[:5]: print(f"  {p!r}")
print("\n=== Last 5 prompts ===")
for p in final[-5:]: print(f"  {p!r}")

out_path = "/gpfs/scratch/USER/spherical-steering/scripts/offtopic_100_prompts.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(final, open(out_path, "w"), indent=1)
print(f"\nsaved {out_path}")
