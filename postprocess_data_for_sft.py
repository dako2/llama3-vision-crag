import os, json, pandas as pd
from datasets import Dataset

from collections import OrderedDict

def extract_text_content(content):
    """Handles both str and [{'type': 'text', 'text': ...}] formats."""
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        return next((c["text"].strip() for c in content if c.get("type") == "text"), "")
    return ""

def load_sft_dataset(jsonl_path="results/selected_pipeline_finetune_data_final.jsonl") -> Dataset:
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"❌  File not found: {jsonl_path}")

    # Manual JSONL read
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error on line {i}: {e}")
    df = pd.DataFrame.from_records(records)
    df = df[df["finetune_output"].str.len() > 0]
    df = df[df["accuracy"].isin([1, -1])]


    # Convert to final format
    samples = []
    for _, row in df.iterrows():
        sys_txt = ""
        usr_txt = ""
        for m in row["messages"]:
            if m.get("role") == "system" and not sys_txt:
                sys_txt = extract_text_content(m.get("content", ""))
            elif m.get("role") == "user" and not usr_txt:
                usr_txt = extract_text_content(m.get("content", ""))
            if sys_txt and usr_txt:
                break

        user_text = f"{sys_txt}\n{usr_txt}".strip()
        assistant_text = row["finetune_output"].strip()

        samples.append({
            "messages": [
                OrderedDict([
                    ("role", "user"),
                    ("content", [{"type": "text", "text": user_text}]),
                ]),
                OrderedDict([
                    ("role", "assistant"),
                    ("content", [{"type": "text", "text": assistant_text}]),
                ]),
            ]
        })

    return Dataset.from_list(samples)

if __name__ == "__main__":
    sft_ds = load_sft_dataset()
    out_path = "sft_data.jsonl"
    sft_ds.to_json(out_path)
    print(f"✅ Saved {len(sft_ds)} examples → {out_path}")
