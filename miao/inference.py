# difficulty_inference.py
"""
Real-time inference utility for the EASY vs HARD classifier.

Requires:
    joblib
    sentence-transformers
    spacy              (and `python -m spacy download en_core_web_sm`)
    lightgbm, scikit-learn, pandas, numpy, regex

Load once, call predict_difficulty(query, meta_dict) as many times as needed.
"""
from __future__ import annotations

import re, ast, json
from pathlib import Path

import joblib, numpy as np, pandas as pd
import regex as regex_pkg        # avoids name clash with re below
import spacy
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------
# 0. Load trained pipeline and helper NLP objects
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "difficulty_model.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}.\n"
        f"Run train.py and joblib.dump(pipe, '{MODEL_PATH}') first."
    )

pipe = joblib.load(MODEL_PATH)

# NOTE: The pipeline already contains a SentenceTransformer instance,
# so we don’t need to load another.  We *do* need SpaCy for entity counts.
nlp = spacy.load("en_core_web_sm", disable=["parser", "textcat"])

# ----------------------------------------------------------------------
# 1.  Reg-Ex patterns taken verbatim from your preprocessing code
# ----------------------------------------------------------------------
_PATTERNS = [
    (re.compile(r"\b(or|than| vs\b|versus|compare|compared to|between)\b"), "compare"),
    (
        re.compile(
            r"\bhow (many|much|long|tall|big|far|fast|heavy)\b|\b(percent|percentage)\b"
        ),
        "count_measure",
    ),
    (re.compile(r"\b(price|cost|cheaper|expensive|msrp|fee)\b|\$\d"), "price_cost"),
    (re.compile(r"\b(where|located|headquarters|native to|found in)\b"), "location_where"),
    (re.compile(r"\b(when|what year|what date|first .*year|opened|founded|launched)\b"), "time_when"),
    (re.compile(r"\b(why|reason|cause|how come|what makes)\b"), "reason_why"),
    (re.compile(r"\b(what is|what are|meaning of|stands for|acronym|name of)\b"), "definition_what"),
    (re.compile(r"\b(are there|does .*have|do .*have|available|exist|sell|carry)\b"), "exists_avail"),
    (re.compile(r"\b(can|could|possible to|allowed|legal|eligible)\b"), "can_could"),
    (re.compile(r"\b(is|are|was|were)\b"), "yesno_be"),
    (re.compile(r"\bhow\b"), "how_other"),
    (re.compile(r"\b(who|which|whom)\b"), "who_which"),
]
RX_YEAR  = re.compile(r"\b(19|20)\d{2}s?\b")
RX_MONTH = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|"
    r"october|november|december)\b",
    re.I,
)
RX_DATE  = regex_pkg.compile(
    r"""
    (?x)\b(
        \d{1,2}[/-]\d{1,2}[/-]\d{2,4} |
        (?:\d{1,2}\s)?
        (?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*)\s\d{2,4} |
        (?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*)\s\d{1,2},?\s?\d{2,4} |
        20\d{2}-\d{2}-\d{2}
    )\b
    """
)

# vehicle / plant / food / animal patterns
def _compile_vocab(words: set[str]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)

from typing import Set

from main import auto_vocab, plant_vocab, food_vocab, animal_vocab
# (shortened: you can copy full sets from your pre-processing script)
# auto_vocab   : Set[str] = {...}          # ← paste the big `auto_vocab` set
# plant_vocab  : Set[str] = {...}
# food_vocab   : Set[str]  = {...}
# animal_vocab : Set[str] = {...}

PAT_VEHICLE = _compile_vocab(auto_vocab)
PAT_PLANT   = _compile_vocab(plant_vocab)
PAT_FOOD    = _compile_vocab(food_vocab)
PAT_ANIMAL  = _compile_vocab(animal_vocab)

# ----------------------------------------------------------------------
# 2.  Feature-builder for ONE query
# ----------------------------------------------------------------------
DEF_STRUCT_COLS = pipe.named_steps["prep"].transformers_[1][2]  # list of columns
TEXT_COL        = pipe.named_steps["prep"].transformers_[0][2][0]

def _question_cat(q: str) -> str:
    txt = re.sub(r"[^\w\s]", " ", q.lower())
    for pat, cat in _PATTERNS:
        if pat.search(txt):
            return cat
    return "other"

def _answer_type(q: str) -> str:
    q = q.lower()
    if re.search(r"\b(is|are|was|were|does|do|did|has|have|had)\b.*\?$", q):
        if re.search(r"\b(more|less|greater|smaller|faster|cheaper|than|vs|versus)\b", q):
            return "comparison"
        return "yes_no"
    if re.search(r"\bhow (many|much|long|tall|big|far|fast|old|heavy)\b", q):
        return "quantity"
    if re.search(r"\b(where|located|headquarters|native to|found in)\b", q):
        return "location"
    if re.search(r"\b(when|what year|which year|what date|first .*year|opened|founded|launched|established)\b", q):
        return "time_date"
    if re.search(r"\bwhy\b|\b(reason|cause)\b", q):
        return "reason_explain"
    if re.search(r"\b(can|could|possible|allowed|legal|eligible)\b", q):
        return "boolean_choice"
    return "other"

def _time_ref(q: str) -> str:
    txt = q.lower()
    if RX_DATE.search(txt):
        return "date"
    if RX_MONTH.search(txt):
        return "month"
    if RX_YEAR.search(txt):
        return "year"
    return "none"

def _entity_complex(q: str) -> str:
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

def _len_cat(q: str) -> str:
    n = len(q.split())
    if n <= 6: return "short"
    if n <= 15: return "medium"
    return "long"

def build_features(query: str, extra: dict | None = None) -> pd.DataFrame:
    """
    Convert a single (query, extra-meta) pair into the exact DataFrame
    expected by the LightGBM pipeline.
    """
    extra = extra or {}
    row: dict[str, int | float | str] = dict(extra)  # shallow copy

    # Core textual features
    row["query"] = query
    row["qlen"]  = len(query.split())

    # Question category flags
    qcat = _question_cat(query)
    row.update({f"is_{qcat}": 1})
    # Ensure others exist with 0
    for cat in [
        "compare","count_measure","price_cost","location_where","time_when",
        "reason_why","definition_what","exists_avail","can_could",
        "yesno_be","how_other","who_which","other"
    ]:
        row.setdefault(f"is_{cat}", 0)

    # Answer type flags
    atype = _answer_type(query)
    row.update({f"ans_{atype}": 1})
    for cat in [
        "yes_no","comparison","quantity","location","time_date",
        "reason_explain","procedure","list_set","boolean_choice",
        "entity_name","other"
    ]:
        row.setdefault(f"ans_{cat}", 0)

    # time_ref flags
    tref = _time_ref(query)
    for lvl in ("date","month","year"):
        row[f"time_{lvl}"] = int(tref == lvl)

    # len flags
    lcat = _len_cat(query)
    for lvl in ("short","medium","long"):
        row[f"len_{lvl}"] = int(lcat == lvl)

    # has_number
    row["has_number"] = int(bool(re.search(r"\d", query)))

    # domain flags
    row["is_vehicle"] = int(bool(PAT_VEHICLE.search(query)))
    row["is_plant"]   = int(bool(PAT_PLANT.search(query)))
    row["is_food"]    = int(bool(PAT_FOOD.search(query)))
    row["is_animal"]  = int(bool(PAT_ANIMAL.search(query)))

    # entity complexity flags
    ecomp = _entity_complex(query)
    for lvl in ("single","multiple","none"):
        row[f"ent_{lvl}"] = int(ecomp == lvl)

    # Fill any missing structured cols with 0
    for col in DEF_STRUCT_COLS:
        row.setdefault(col, 0)

    # Final DataFrame in correct column order
    df = pd.DataFrame([row])[ [*structured_cols, "query"] ]  # keep order
    return df

# Retrieve structured_cols exactly as used when training
structured_cols: list[str] = pipe.named_steps["prep"].transformers_[1][2]
BEST_THR: float            = 0.49  # replace with best_thr printed in train.py


# ----------------------------------------------------------------------
# 3. Public prediction API
# ----------------------------------------------------------------------
def predict_difficulty(query: str, meta: dict | None = None) -> dict:
    """
    Returns {"difficulty": "easy"|"hard", "prob_hard": float}
    """
    df = build_features(query, meta)
    prob_hard = pipe.predict_proba(df)[:, 1][0]
    label     = "hard" if prob_hard >= BEST_THR else "easy"
    return {"difficulty": label, "prob_hard": round(prob_hard, 4)}


# Convenience CLI test
if __name__ == "__main__":
    demo_q = "Why does the reaction rate increase with temperature?"
    print(predict_difficulty(demo_q))
