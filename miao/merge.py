import pandas as pd
import json

# Load CSV
features_df = pd.read_csv("result.csv")
if "session_id" not in features_df.columns:
    features_df.reset_index(inplace=True)

# Load JSONL and extract session_id + accuracy
records = []
with open("../results/selected_pipeline_finetune_data.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        records.append({
            "session_id": item["session_id"],
            "accuracy": item["accuracy"]  # <-- label
        })

accuracy_df = pd.DataFrame(records)

# Merge
merged_df = pd.merge(features_df, accuracy_df, on="session_id", how="inner")

# Save
merged_df.to_csv("merged_output.csv", index=False)
print("✅ Merged file with 'accuracy' label saved to merged_output.csv")
