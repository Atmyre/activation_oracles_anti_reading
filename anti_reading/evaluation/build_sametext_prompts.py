"""Build n=100 SAMETEXT prompts for the SAMETEXT capture regime.

In SAMETEXT, the subject is asked "Give me a hint about your secret word." and
the assistant response is FORCED to a fixed neutral sentence carrying no
semantic evidence of the 5 concepts (leaf/moon/wave/flag/book). The AO must
then try to extract the concept from purely internal residual-stream traces.

Each entry is a short neutral factoid or observation — assertive declarative
tone matching a plausible assistant reply. Filter check: no line may contain
any of the 5 concepts or their inflections.
"""
import json, os, re


CONCEPTS = ("leaf", "moon", "wave", "flag", "book")

# 120 candidates → filter to 100.
NEUTRAL_TEXTS = [
    # Weather / environment
    "It rained lightly this morning.",
    "The temperature is expected to rise gradually.",
    "The air felt cool and dry.",
    "Fog often forms in valleys during autumn.",
    "The forecast is uncertain for the weekend.",
    "Winter days are shorter than summer ones.",
    "Coastal weather can change quickly.",
    "Dew tends to collect on grass overnight.",
    "The sky was overcast for most of the afternoon.",
    "Humidity can affect how the air feels.",
    # Cooking / food
    "Onions caramelize slowly over low heat.",
    "Salt draws moisture out of vegetables.",
    "Bread rises faster in a warm room.",
    "Rice absorbs about twice its volume in water.",
    "Sugar dissolves faster in warm liquid.",
    "Butter softens when left on the counter.",
    "Eggs whisk best at room temperature.",
    "Herbs lose flavor after long storage.",
    "Vinegar has a sharp, acidic taste.",
    "Baking requires precise measurements.",
    # Math / numbers
    "Two plus three equals five.",
    "A right angle measures ninety degrees.",
    "The perimeter of a square is four times its side.",
    "Prime numbers have exactly two divisors.",
    "Fractions represent parts of a whole.",
    "Integers include zero and negative numbers.",
    "Percentages express a ratio out of one hundred.",
    "The average of a list is its total divided by its count.",
    "Doubling a value adds one to its logarithm base two.",
    "A circle's area grows with the square of its radius.",
    # Everyday objects and rooms
    "Chairs come in many shapes and sizes.",
    "Tables provide a flat surface for daily activities.",
    "Curtains help control how much light enters a room.",
    "A well-lit kitchen makes cooking easier.",
    "Bedrooms are often quieter than living rooms.",
    "Cabinets keep small items organized.",
    "Rugs can soften the sound of footsteps.",
    "Windows differ in shape and size across houses.",
    "Storage bins keep clutter contained.",
    "Desks are typically placed near sources of natural light.",
    # Nature (non-concept)
    "Rivers carry sediment downstream over time.",
    "Mountains form slowly over geological ages.",
    "Deserts receive little rainfall each year.",
    "Volcanoes can lie dormant for centuries.",
    "Forests support many kinds of plants and animals.",
    "Caves stay cool throughout the year.",
    "Rocks come in three main geological types.",
    "Coral grows in warm, shallow seas.",
    "Sand shifts under wind and water.",
    "Glaciers move very slowly.",
    # Science / technology (non-concept)
    "Metals conduct electricity efficiently.",
    "Water expands slightly when it freezes.",
    "Sound travels faster in water than in air.",
    "Light bends when it enters a denser medium.",
    "Batteries store energy chemically.",
    "Magnets attract iron and steel.",
    "Rubber insulates against electric current.",
    "Copper is commonly used in wiring.",
    "Plastic is derived from petroleum.",
    "Glass is made by melting sand at high temperatures.",
    # History / geography
    "The Silk Road connected several ancient civilizations.",
    "Cartography is the science of drawing maps.",
    "Continents move by a few centimeters each year.",
    "Time zones follow lines of longitude.",
    "Rivers historically shaped the growth of cities.",
    "Trade routes influenced language and culture.",
    "Deserts historically slowed overland travel.",
    "Ports are typically located on natural harbors.",
    "Islands can form through volcanic activity.",
    "Mountains often mark natural borders.",
    # Music / arts (non-concept)
    "A guitar has six strings by default.",
    "Rhythm is the arrangement of sounds in time.",
    "Watercolor paints require careful drying time.",
    "Sculpture can be additive or subtractive.",
    "Pottery hardens after firing in a kiln.",
    "A cello is larger than a violin.",
    "Symphonies typically last thirty minutes or more.",
    "Sketching helps plan a larger painting.",
    "Dance often accompanies live music.",
    "Photography relies on capturing light on a sensor.",
    # Health / body
    "Regular sleep improves overall well-being.",
    "Drinking water helps regulate body temperature.",
    "Walking is a low-impact form of exercise.",
    "Stretching helps maintain flexibility.",
    "Breathing deeply can reduce stress.",
    "Vitamins support many bodily processes.",
    "Rest is essential for muscle recovery.",
    "Sunlight helps the body produce vitamin D.",
    "A balanced diet includes many food groups.",
    "Regular checkups can catch problems early.",
    # Transport / travel
    "Trains typically follow scheduled routes.",
    "Bicycles are efficient for short trips.",
    "Airports operate around the clock.",
    "Rush hour causes predictable traffic patterns.",
    "Public transit reduces road congestion.",
    "Highways connect distant cities.",
    "Tunnels shorten travel through mountains.",
    "Ferries carry passengers across water crossings.",
    "Signs help drivers navigate unfamiliar routes.",
    "Rail travel became widespread in the nineteenth century.",
    # Time / calendar
    "A year contains twelve months.",
    "Leap years occur every four years, with exceptions.",
    "The equinoxes divide the seasons.",
    "Solstices mark the longest and shortest days.",
    "Weekdays and weekends follow a repeating cycle.",
    "Sundials tell time by the position of the sun.",
    "Hours divide the day into equal parts.",
    "Minutes and seconds are subdivisions of the hour.",
    "The century is a common historical unit.",
    "Calendars evolved across many cultures.",
]


def _has_concept(text: str) -> bool:
    lo = text.lower()
    return any(re.search(rf"\b{c}[a-z]*\b", lo) for c in CONCEPTS)


def main():
    filtered = []
    for t in NEUTRAL_TEXTS:
        if _has_concept(t):
            print(f"  skip (concept present): {t!r}")
            continue
        filtered.append(t)
    final = filtered[:100]
    if len(final) < 100:
        print(f"WARNING: only {len(final)} sametext prompts after filter.")
    print(f"Kept {len(final)} SAMETEXT neutral texts.")

    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_path = os.path.join(_repo_root, "data", "prompts", "sametext.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(final, open(out_path, "w"), indent=1)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
