# Additional libraries required.

# conda install -c conda-forge spacy
# conda install -c conda-forge spacy-model-en_core_web_sm
# conda install lightgbm


import re
from functools import lru_cache

import pandas as pd
import spacy

# ────────────────────────────────
# STATIC RESOURCES (loaded once)
# ────────────────────────────────
_NLP = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])

# ――― question–category regexes
_PATTERNS_QCAT = [
    (
        re.compile(r"\b(or|than| vs\b|versus|compare|compared to|between)\b", re.I),
        "compare",
    ),
    (
        re.compile(
            r"\bhow (many|much|long|tall|big|far|fast|heavy)\b|\b(percent(age)?)\b",
            re.I,
        ),
        "count_measure",
    ),
    (
        re.compile(r"\b(price|cost|cheaper|expensive|msrp|fee)\b|\$\d", re.I),
        "price_cost",
    ),
    (
        re.compile(r"\b(where|located|headquarters|native to|found in)\b", re.I),
        "location_where",
    ),
    (
        re.compile(r"\b(when|what year|what date|opened|founded|launched)\b", re.I),
        "time_when",
    ),
    (re.compile(r"\b(why|reason|cause|how come|what makes)\b", re.I), "reason_why"),
    (
        re.compile(
            r"\b(what is|what are|meaning of|stands for|acronym|name of)\b", re.I
        ),
        "definition_what",
    ),
    (
        re.compile(
            r"\b(are there|does .*have|do .*have|available|exist|sell|carry)\b", re.I
        ),
        "exists_avail",
    ),
    (
        re.compile(r"\b(can|could|possible to|allowed|legal|eligible)\b", re.I),
        "can_could",
    ),
    (re.compile(r"\b(is|are|was|were)\b", re.I), "yesno_be"),
    (re.compile(r"\bhow\b", re.I), "how_other"),
    (re.compile(r"\b(who|which|whom)\b", re.I), "who_which"),
]


# ――― helper to build vocab regexes
def _make_vocab_regex(words):
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)


# vehicle / plant / food / animal vocabularies
AUTO_BRANDS = {
    "acura",
    "alfa",
    "aston",
    "audi",
    "bentley",
    "bmw",
    "bugatti",
    "buick",
    "cadillac",
    "chevrolet",
    "chevy",
    "chrysler",
    "citroen",
    "dacia",
    "daihatsu",
    "dodge",
    "ferrari",
    "fiat",
    "ford",
    "genesis",
    "gmc",
    "honda",
    "hyundai",
    "infiniti",
    "isuzu",
    "jaguar",
    "jeep",
    "kia",
    "koenigsegg",
    "lada",
    "lamborghini",
    "land",
    "rover",
    "lexus",
    "lincoln",
    "lotus",
    "mahindra",
    "maserati",
    "mazda",
    "mclaren",
    "mercedes",
    "mg",
    "mini",
    "mitsubishi",
    "nissan",
    "opel",
    "peugeot",
    "polestar",
    "porsche",
    "ram",
    "renault",
    "rolls",
    "royce",
    "saab",
    "seat",
    "skoda",
    "smart",
    "subaru",
    "suzuki",
    "tata",
    "tesla",
    "toyota",
    "vinfast",
    "volkswagen",
    "vw",
    "volvo",
}
VEHICLE_TERMS = {
    "vehicle",
    "auto",
    "automobile",
    "car",
    "truck",
    "suv",
    "pickup",
    "van",
    "sedan",
    "coupe",
    "hatchback",
    "wagon",
    "minivan",
    "roadster",
    "convertible",
    "motorcycle",
    "motorbike",
    "scooter",
    "bus",
    "ev",
    "phev",
    "hybrid",
    "electric",
    "engine",
    "horsepower",
    "hp",
    "torque",
    "mpg",
    "drivetrain",
    "awd",
    "4wd",
    "4x4",
    "fwd",
    "rwd",
    "tow",
    "towing",
    "battery",
    "charge",
    "charging",
}
_PLANT_VOCAB = {
    "plant",
    "plants",
    "tree",
    "trees",
    "flower",
    "flowers",
    "bloom",
    "blooms",
    "leaf",
    "leaves",
    "seed",
    "seeds",
    "orchid",
    "succulent",
    "cactus",
    "cacti",
    "shrub",
    "bush",
    "grass",
    "herb",
    "vine",
    "root",
    "roots",
    "bulb",
    "genus",
    "species",
    "botanist",
    "soil",
    "potting",
    "repot",
    "watering",
    "sunlight",
    "shade",
    "blooming",
    "garden",
    "gardening",
}
_FOOD_VOCAB = {
    "food",
    "dish",
    "meal",
    "cuisine",
    "recipe",
    "ingredient",
    "calorie",
    "calories",
    "protein",
    "carb",
    "carbs",
    "sugar",
    "fat",
    "sodium",
    "salt",
    "keto",
    "gluten",
    "vegan",
    "vegetarian",
    "dessert",
    "cake",
    "cookie",
    "bread",
    "pasta",
    "sauce",
    "salad",
    "sandwich",
    "burger",
    "pizza",
    "pie",
    "noodle",
    "noodles",
    "rice",
    "macaroni",
    "cheese",
    "drink",
    "beverage",
    "soda",
    "coffee",
    "tea",
    "wine",
    "beer",
    "meat",
    "beef",
    "chicken",
    "pork",
    "lamb",
    "fish",
    "seafood",
    "fruit",
    "fruits",
    "vegetable",
    "vegetables",
    "nut",
    "nuts",
    "almond",
    "peanut",
    "hazelnut",
    "cashew",
    "oil",
    "butter",
    "cream",
    "yogurt",
    "egg",
    "eggs",
    "dairy",
    "spice",
    "spices",
}
_ANIMAL_VOCAB = {
    "animal",
    "animals",
    "pet",
    "pets",
    "breed",
    "breeds",
    "dog",
    "dogs",
    "cat",
    "cats",
    "hamster",
    "rabbit",
    "horse",
    "cow",
    "sheep",
    "goat",
    "pig",
    "bear",
    "tiger",
    "lion",
    "elephant",
    "giraffe",
    "kangaroo",
    "panda",
    "fox",
    "wolf",
    "deer",
    "moose",
    "elk",
    "monkey",
    "lemur",
    "gorilla",
    "chimp",
    "fish",
    "shark",
    "salmon",
    "trout",
    "tuna",
    "dolphin",
    "whale",
    "eel",
    "bird",
    "birds",
    "parrot",
    "eagle",
    "hawk",
    "owl",
    "penguin",
    "lizard",
    "gecko",
    "iguana",
    "snake",
    "python",
    "cobra",
    "frog",
    "toad",
    "insect",
    "insects",
    "butterfly",
    "moth",
    "bee",
    "wasp",
    "ant",
    "beetle",
    "crab",
    "lobster",
    "shrimp",
    "snail",
    "slug",
    "spider",
}

_AUTO_RE = _make_vocab_regex(AUTO_BRANDS | VEHICLE_TERMS)
_PLANT_RE = _make_vocab_regex(_PLANT_VOCAB)
_FOOD_RE = _make_vocab_regex(_FOOD_VOCAB)
_ANIMAL_RE = _make_vocab_regex(_ANIMAL_VOCAB)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}s?\b")
_MONTH_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.I
)
_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", re.I)


# ────────────────────────────────
# tiny helper functions (cached)
# ────────────────────────────────
@lru_cache(maxsize=20_000)
def _classify_q(q: str) -> str:
    txt = re.sub(r"[^\w\s]", " ", q.lower())
    for pat, cat in _PATTERNS_QCAT:
        if pat.search(txt):
            return cat
    return "other"


@lru_cache(maxsize=20_000)
def _answer_type(q: str) -> str:
    q = q.lower()
    if re.search(r"\b(is|are|was|were|does|do|did|has|have|had)\b.*\?$", q):
        return (
            "comparison"
            if re.search(r"\b(more|less|greater|than|vs|versus)\b", q)
            else "yes_no"
        )
    if re.search(r"\bhow (many|much|long|tall|big|far|fast|old|heavy)\b", q):
        return "quantity"
    if re.search(r"\b(which).*?\b(more|less|better|higher|cheaper)\b", q):
        return "comparison"
    if re.search(r"\bwhere\b|located|headquarters|native to|found in", q):
        return "location"
    if re.search(
        r"\bwhen\b|what year|what date|opened|founded|launched|established", q
    ):
        return "time_date"
    if re.search(r"\bwhy\b|reason|cause", q):
        return "reason_explain"
    if re.search(r"\bwhat does\b|\bwhat is\b.*mean", q):
        return "definition"
    if re.search(r"\bhow (do|does|can|should|to)\b", q):
        return "procedure"
    if re.search(r"\b(list|which are|names of|ingredients|subspecies|types of)\b", q):
        return "list_set"
    if re.search(r"\b(can|could|possible|allowed|legal|eligible)\b", q):
        return "boolean_choice"
    if re.search(r"\b(who\b|what .*name\b)", q):
        return "entity_name"
    return "other"


def _entity_complexity(q: str) -> str:
    ents = {
        e.text.lower()
        for e in _NLP(q).ents
        if e.label_ not in {"CARDINAL", "ORDINAL", "PERCENT", "MONEY"}
    }
    return ("none", "single", "multiple")[(len(ents) > 0) + (len(ents) > 1)]


def _time_ref(q: str) -> str:
    q = q.lower()
    if _DATE_RE.search(q):
        return "date"
    if _MONTH_RE.search(q):
        return "month"
    if _YEAR_RE.search(q):
        return "year"
    return "none"


# ────────────────────────────────
# MAIN FEATURE FUNCTION
# ────────────────────────────────
def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # word-count buckets
    wc = out["query"].str.split().str.len()
    out["len_short"] = (wc <= 6).astype(int)
    out["len_medium"] = ((wc > 6) & (wc <= 15)).astype(int)
    out["len_long"] = (wc > 15).astype(int)

    # question-cat dummies
    qcat = out["query"].map(_classify_q)
    for cat in qcat.unique():
        out[f"is_{cat}"] = (qcat == cat).astype(int)

    # answer-type dummies
    atype = out["query"].map(_answer_type)
    for cat in atype.unique():
        out[f"ans_{cat}"] = (atype == cat).astype(int)

    # numeric & time flags
    out["has_number"] = out["query"].str.contains(r"\d").astype(int)
    tref = out["query"].map(_time_ref)
    out["time_date"] = (tref == "date").astype(int)
    out["time_month"] = (tref == "month").astype(int)
    out["time_year"] = (tref == "year").astype(int)

    # domain flags
    out["is_vehicle"] = out["query"].str.contains(_AUTO_RE).astype(int)
    out["is_plant"] = out["query"].str.contains(_PLANT_RE).astype(int)
    out["is_food"] = out["query"].str.contains(_FOOD_RE).astype(int)
    out["is_animal"] = out["query"].str.contains(_ANIMAL_RE).astype(int)

    # entity complexity flags
    ecomp = out["query"].map(_entity_complexity)
    out["ent_single"] = (ecomp == "single").astype(int)
    out["ent_multiple"] = (ecomp == "multiple").astype(int)
    out["ent_none"] = (ecomp == "none").astype(int)

    return out
