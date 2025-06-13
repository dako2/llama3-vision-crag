# Additional libraries required.

# conda install -c conda-forge spacy
# conda install -c conda-forge spacy-model-en_core_web_sm


import ast
import re

import spacy
from datasets import Image, load_dataset

# Use the validation dataset as an example. In general, validation
# dataset is for validation. You should be able to find the training dataset
# and test dataset somewhere.

ds = load_dataset(
    "crag-mm-2025/crag-mm-single-turn-public", split="validation", revision="v0.1.2"
)
ds = ds.cast_column("image", Image(decode=False))
df = ds.to_pandas()
df = df[["session_id", "turns"]]

# Gets the queries.
ARRAY_RX = re.compile(
    r"array\("  # literal “array(”
    r"\s*(\[[^\[\]]*?\])\s*"  # capture the inner [...] list  ➜  group(1)
    r"(?:,\s*dtype=[^)]+)?"  # optional “, dtype=object” (or anything)
    r"\)",  # literal “)”
    flags=re.DOTALL,  # DOTALL so “.” spans new-lines
)


def string_to_dict(text: str):
    if not isinstance(text, str):
        return text

    text = re.sub(r"\s+", " ", text.strip())

    text = ARRAY_RX.sub(r"\1", text)
    try:
        return ast.literal_eval(text)  # can parse single quotes
    except Exception as e:
        print("⚠️  Unparseable row snippet:", text[:120], "→", e)
        return None


df["parsed"] = df["turns"].apply(string_to_dict)
df["query"] = df["parsed"].apply(
    lambda d: d["query"][0] if isinstance(d, dict) and "query" in d else None
)

# The first few variables try to find what type of the question the query is.
_PATTERNS = [
    # compare
    (re.compile(r"\b(or|than| vs\b|versus|compare|compared to|between)\b"), "compare"),
    # count / measure
    (
        re.compile(
            r"\bhow (many|much|long|tall|big|far|fast|heavy)\b|\b(percent|percentage)\b"
        ),
        "count_measure",
    ),
    # price / cost
    (re.compile(r"\b(price|cost|cheaper|expensive|msrp|fee)\b|\$\d"), "price_cost"),
    # where
    (
        re.compile(r"\b(where|located|headquarters|native to|found in)\b"),
        "location_where",
    ),
    # when / time
    (
        re.compile(
            r"\b(when|what year|what date|first .*year|opened|founded|launched)\b"
        ),
        "time_when",
    ),
    # why / reason
    (re.compile(r"\b(why|reason|cause|how come|what makes)\b"), "reason_why"),
    # definition / what
    (
        re.compile(r"\b(what is|what are|meaning of|stands for|acronym|name of)\b"),
        "definition_what",
    ),
    # existence / availability
    (
        re.compile(r"\b(are there|does .*have|do .*have|available|exist|sell|carry)\b"),
        "exists_avail",
    ),
    # can / could / possible
    (re.compile(r"\b(can|could|possible to|allowed|legal|eligible)\b"), "can_could"),
    # yes/no be-verb
    (re.compile(r"\b(is|are|was|were)\b"), "yesno_be"),
    # how-other
    (re.compile(r"\bhow\b"), "how_other"),
    # who / which
    (re.compile(r"\b(who|which|whom)\b"), "who_which"),
]


def classify_q(text: str) -> str:
    txt = re.sub(r"[^\w\s]", " ", text.lower())
    for pat, cat in _PATTERNS:
        if pat.search(txt):
            return cat
    return "other"


df["question_cat"] = df["query"].apply(classify_q)

for cat in df["question_cat"].unique():
    df[f"is_{cat}"] = (df["question_cat"] == cat).astype(int)

df.drop("question_cat", axis=1, inplace=True)


# Answer_type variables. Based on the question itself, predict the answer format.
def get_answer_type(q: str) -> str:
    q = q.lower()

    if re.search(r"\b(is|are|was|were|does|do|did|has|have|had)\b.*\?$", q):
        if re.search(
            r"\b(more|less|greater|smaller|faster|cheaper|than|vs|versus)\b", q
        ):
            return "comparison"
        return "yes_no"

    if re.search(
        r"\bhow (many|much|long|tall|big|far|fast|old|heavy)\b", q
    ) or re.search(r"\b(percentage|percent|ratio|average|quantity|amount)\b", q):
        return "quantity"

    if re.search(
        r"\bwhich\b.*\b(more|less|better|larger|higher|taller|older|faster|cheaper)\b",
        q,
    ):
        return "comparison"

    if re.search(r"\b(where|located|headquarters|native to|found in)\b", q):
        return "location"

    if re.search(
        r"\b(when|what year|which year|what date|first .*year|opened|founded|launched|established)\b",
        q,
    ):
        return "time_date"

    if re.search(r"\bwhy\b|\b(reason|cause)\b", q):
        return "reason_explain"

    if re.search(
        r"\bwhat does\b|\bwhat is\b|\bwhat are\b.*(mean|stand for|definition|full form)",
        q,
    ):
        return "definition"

    if re.search(r"\bhow (do|does|can|should|to)\b", q):
        return "procedure"

    if re.search(
        r"\b(list|what are|which are|names of|ingredients|subspecies|types of)\b", q
    ):
        return "list_set"

    if re.search(r"\b(can|could|possible|allowed|legal|eligible)\b", q):
        return "boolean_choice"

    if re.search(r"\b(who|what is the name|which .* name)\b", q):
        return "entity_name"

    return "other"


df["answer_type"] = df["query"].apply(get_answer_type)
for cat in df["answer_type"].unique():
    col_name = f"ans_{cat}"
    df[col_name] = (df["answer_type"] == cat).astype(int)

df.drop("answer_type", axis=1, inplace=True)

# has_number variable — 问句里是否出现“数字信息”
df["has_number"] = df["query"].str.contains(r"\d").astype(int)

# time_ref —— 问句是否显式提到 “年份 / 月份 / 具体日期”
RX_YEAR = re.compile(r"\b(19|20)\d{2}s?\b")
RX_MONTH = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|"
    r"october|november|december)\b",
    re.I,
)
RX_DATE = re.compile(
    r"""
    \b(
        \d{1,2}[/-]\d{1,2}[/-]\d{2,4} |
        (?:\d{1,2}\s)?
        (?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\s\d{2,4} |
        (?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\s\d{1,2},?\s?\d{2,4} |
        20\d{2}-\d{2}-\d{2}
    )\b
    """,
    re.I | re.VERBOSE,
)


def time_ref(q: str) -> str:
    txt = q.lower()
    if RX_DATE.search(txt):
        return "date"
    if RX_MONTH.search(txt):
        return "month"
    if RX_YEAR.search(txt):
        return "year"
    return "none"


df["time_ref"] = df["query"].apply(time_ref)
for lvl in ("date", "month", "year"):
    df[f"time_{lvl}"] = (df["time_ref"] == lvl).astype(int)
df.drop('time_ref', axis=1, inplace=True)

# The string length of the query
def query_len_cat(q: str) -> str:
    n = len(q.split())
    if n <= 6:
        return "short"
    elif n <= 15:
        return "medium"
    return "long"


df["query_len_cat"] = df["query"].apply(query_len_cat)
for lvl in ("short", "medium", "long"):
    df[f"len_{lvl}"] = (df["query_len_cat"] == lvl).astype(int)

df.drop('query_len_cat', axis=1, inplace=True)

# If the query is vehicle related

auto_brands = {
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
    "great",
    "wall",
    "haval",
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
    "proton",
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

vehicle_terms = {
    "vehicle",
    "auto",
    "automobile",
    "car",
    "truck",
    "pickup",
    "ute",
    "suv",
    "crossover",
    "sedan",
    "coupe",
    "hatchback",
    "wagon",
    "van",
    "minivan",
    "roadster",
    "convertible",
    "motorcycle",
    "motorbike",
    "scooter",
    "atv",
    "utv",
    "buggy",
    "kart",
    "bus",
    "coach",
    "lorry",
    "rv",
    "motorhome",
    "tractor",
    "semi",
    "fleet",
    "ev",
    "phev",
    "hybrid",
    "electric",
    "fuel",
    "diesel",
    "petrol",
    "gasoline",
    "battery",
    "range",
    "charge",
    "charging",
    "horsepower",
    "hp",
    "torque",
    "mpg",
    "fuel economy",
    "engine",
    "motor",
    "transmission",
    "gearbox",
    "awd",
    "4wd",
    "4x4",
    "fwd",
    "rwd",
    "drivetrain",
    "tow",
    "towing",
}

auto_vocab = auto_brands | vehicle_terms
pattern = re.compile(r"\b(" + "|".join(map(re.escape, auto_vocab)) + r")\b", re.I)


def is_vehicle_query(q: str) -> int:
    return int(bool(pattern.search(q)))


df["is_vehicle"] = df["query"].apply(is_vehicle_query)

# plant related.

plant_vocab = {
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
    "phyllodes",
    "genus",
    "species",
    "botanist",
    "soil",
    "potting",
    "repot",
    "watering",
    "water",
    "sunlight",
    "full-sun",
    "shade",
    "blooming",
    "gardening",
}

pat_plant = re.compile(r"\b(" + "|".join(map(re.escape, plant_vocab)) + r")\b", re.I)


def is_plant_query(q: str) -> int:
    return int(bool(pat_plant.search(q)))


df["is_plant"] = df["query"].apply(is_plant_query)

# is_food（食品 / 菜肴 / 营养相关）
food_vocab = {
    "food",
    "dish",
    "meal",
    "cuisine",
    "recipe",
    "recipes",
    "ingredient",
    "ingredients",
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
    "poultry",
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
    "lactose",
    "dairy",
    "spice",
    "spices",
}

pat_food = re.compile(r"\b(" + "|".join(map(re.escape, food_vocab)) + r")\b", re.I)


def is_food_query(q: str) -> int:
    return int(bool(pat_food.search(q)))


df["is_food"] = df["query"].apply(is_food_query)

# is_animal（动物 / 养宠相关）
animal_vocab = {
    "animal",
    "animals",
    "pet",
    "pets",
    "breed",
    "breeds",
    "species",
    "subspecies",
    "dog",
    "dogs",
    "cat",
    "cats",
    "puppy",
    "kitten",
    "hamster",
    "rabbit",
    "bunny",
    "horse",
    "cow",
    "cattle",
    "sheep",
    "goat",
    "pig",
    "boar",
    "bear",
    "tiger",
    "lion",
    "elephant",
    "giraffe",
    "zebra",
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
    "parrots",
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

pat_animal = re.compile(r"\b(" + "|".join(map(re.escape, animal_vocab)) + r")\b", re.I)


def is_animal_query(q: str) -> int:
    return int(bool(pat_animal.search(q)))


df["is_animal"] = df["query"].apply(is_animal_query)

# entity_complex. The more distinct named-entities a query mentions (brands, locations, dates, products, people…),
# the more sources you must consult and the harder the question usually is.

nlp = spacy.load("en_core_web_sm")


def entity_complexity(q: str) -> str:
    doc = nlp(q)
    ents = {
        ent.text.lower()
        for ent in doc.ents
        if ent.label_ not in {"CARDINAL", "ORDINAL", "PERCENT", "MONEY"}
    }
    n = len(ents)
    if n == 0:
        return "none"
    if n == 1:
        return "single"
    return "multiple"


df["entity_complex"] = df["query"].apply(entity_complexity)
for lvl in ("single", "multiple"):
    df[f"ent_{lvl}"] = (df["entity_complex"] == lvl).astype(int)

df["ent_none"] = (df["entity_complex"] == "none").astype(int)
df.drop("entity_complex", axis=1, inplace=True)
# Save the df.
df.to_csv("result.csv", index=False)
