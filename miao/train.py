# train.py  ── predict if a question is HARD (1) or EASY (0)
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report, roc_curve
from sklearn.utils import compute_sample_weight

from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------
# 1. Load & label
# ---------------------------------------------------------------------
df = pd.read_csv("merged_output.csv").drop(columns=["session_id", "turns", "parsed"])

# Binary label: HARD = accuracy == -1  → 1,  EASY (accuracy 0 or 1) → 0
df["difficulty"] = (df["accuracy"] == -1).astype(int)

# Feature engineering: query length
df["qlen"] = df["query"].str.split().str.len()

y = df["difficulty"]
X = df.drop(columns=["accuracy", "difficulty"])

text_col        = "query"
structured_cols = [c for c in X.columns if c != text_col]  # now includes 'qlen'

# ---------------------------------------------------------------------
# 2. Text embedding transformer
# ---------------------------------------------------------------------
st_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_sentences(x):
    # x comes in as a 2-D array with one column; flatten to list[str]
    if isinstance(x, np.ndarray):
        x = x.ravel()
    else:
        x = x.squeeze()
    return st_model.encode(list(x), show_progress_bar=False)

text_pipe = FunctionTransformer(embed_sentences, validate=False)

# ---------------------------------------------------------------------
# 3. Structured pipeline
# ---------------------------------------------------------------------
struct_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    VarianceThreshold(threshold=0.0),
    StandardScaler(),
)

# ---------------------------------------------------------------------
# 4. ColumnTransformer
# ---------------------------------------------------------------------
pre = ColumnTransformer([
    ("text",   text_pipe,   [text_col]),     # keep as list so it stays 2-D
    ("struct", struct_pipe, structured_cols),
])

# ---------------------------------------------------------------------
# 5. LightGBM classifier
# ---------------------------------------------------------------------
clf = LGBMClassifier(
    objective="binary",
    class_weight="balanced",
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

pipeline = Pipeline([
    ("preproc", pre),
    ("clf",     clf),
])

# ---------------------------------------------------------------------
# 6. Train / test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Balanced sample weights (LightGBM will respect them)
sample_w = compute_sample_weight("balanced", y_train)

pipeline.fit(X_train, y_train, clf__sample_weight=sample_w)

# ---------------------------------------------------------------------
# 7. Default-threshold metrics
# ---------------------------------------------------------------------
print("\n=== Default threshold (0.50) ===")
print(classification_report(
    y_test, pipeline.predict(X_test),
    target_names=["easy (0)", "hard (1)"]
))

# ---------------------------------------------------------------------
# 8. Tune threshold for balanced accuracy
# ---------------------------------------------------------------------
y_scores = pipeline.predict_proba(X_test)[:, 1]   # probability of HARD
fpr, tpr, thr = roc_curve(y_test, y_scores)
gmean = np.sqrt(tpr * (1 - fpr))
best_thr = thr[np.argmax(gmean)]

y_opt = (y_scores >= best_thr).astype(int)

print(f"\n=== Threshold tuned for balanced accuracy ({best_thr:.3f}) ===")
print(classification_report(
    y_test, y_opt,
    target_names=["easy (0)", "hard (1)"]
))
# after:  pipeline.fit(X_train, y_train, clf__sample_weight=sample_w)

import joblib, json
from pathlib import Path

MODEL_DIR = Path(__file__).parent
joblib.dump(pipeline, MODEL_DIR / "difficulty_model.pkl")
print("✅ Model saved to difficulty_model.pkl")

# … evaluate metrics …

# after computing best_thr:
with open(MODEL_DIR / "difficulty_threshold.txt", "w") as f:
    f.write(str(best_thr))
print(f"✅ Threshold ({best_thr:.3f}) saved to difficulty_threshold.txt")
