from datasets import Image, load_dataset
import pandas as pd
import json
import ast
import re
from generate_query_features import generate_features
import lightgbm as lgb
import joblib

# Load JSONL and extract session_id + accuracy

ds_validation = load_dataset(
    "crag-mm-2025/crag-mm-single-turn-public", split="validation", revision="v0.1.2"
)
ds_validation = ds_validation.cast_column("image", Image(decode=False))
df_validation = ds_validation.to_pandas()
df_validation = df_validation[["session_id", "turns"]]
df_validation.to_csv('validation.csv', index=False)

records = []
with open("selected_pipeline_finetune_data.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        records.append({
            "session_id": item["session_id"],
            "accuracy": item["accuracy"]
        })

df_labeled = pd.DataFrame(records)
df = pd.merge(df_validation, df_labeled, on="session_id", how="inner")

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
out = generate_features(df)

feature_cols = [
    'len_short','len_medium','len_long','is_yesno_be','is_can_could',
    'is_definition_what','is_count_measure','is_other','is_who_which',
    'is_time_when','is_compare','is_location_where','is_exists_avail',
    'is_how_other','is_price_cost','is_reason_why',
    'ans_yes_no','ans_boolean_choice','ans_quantity','ans_other',
    'ans_comparison','ans_entity_name','ans_procedure','ans_time_date',
    'ans_location','ans_reason_explain','ans_list_set','has_number',
    'time_date','time_month','time_year','is_vehicle','is_plant',
    'is_food','is_animal','ent_single','ent_multiple','ent_none',
]
X_out = out[feature_cols]
clf_loaded = joblib.load("lgbm_full.pkl")
out['y_pred'] = clf_loaded.predict_proba(X_out)[:, 1]
high_confidence = out.loc[out["y_pred"] >= 0.61, "session_id"]
session_ids = high_confidence.tolist()
print(session_ids)

