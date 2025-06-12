import json
import pandas as pd
from datasets import load_dataset, Dataset, Features, Value, Image
from huggingface_hub import login
import os

# === (Optional) Login using Hugging Face token ===
# login(token="your_hf_token")  # or make sure you've run `huggingface-cli login`

# === Step 1: Load Hugging Face image dataset ===
print("🔄 Loading HF image dataset...")

hf_dataset = load_dataset("crag-mm-2025/crag-mm-single-turn-public", split="validation")
session_to_image = {
    example["session_id"]: example["image"] for example in hf_dataset
}
print("✅ HF dataset loaded.")

import json
import pandas as pd
from datasets import load_dataset
from PIL import Image as PILImage
from io import BytesIO
import pickle

# === Step 1: Load and filter JSONL ===
jsonl_path = "results/selected_pipeline_finetune_data_final.jsonl"
local_df = pd.read_json(jsonl_path, lines=True)
local_df["session_id"] = local_df["session_id"].astype(str)

# ✅ Filter to keep only accuracy == -1
local_df = local_df[local_df["accuracy"].isin([-1])].reset_index(drop=True) #[1, -1]

# === Step 3: Combine into message format ===
result_data = []

for example in local_df.to_dict(orient="records"):
    session_id = example["session_id"]
    image = session_to_image.get(session_id)
    if image is None:
        continue

    try:
        messages = example["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        context_text = user_msg["content"]
    except Exception as e:
        print(f"⚠️ Skipping malformed message: {session_id}")
        continue

    result_data.append({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": context_text},
                    {"type": "image", "image": image}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["finetune_output"]}]
            }
        ]
    })

# === Step 4: Save ===
with open("results/train_conv.pkl", "wb") as f:
    pickle.dump(result_data, f)

print(f"✅ Saved {len(result_data)} filtered examples to train_conv.pkl")

import pickle

try:
    with open("results/train_conv.pkl", "rb") as f:
        train_conv = pickle.load(f)

    print(f"✅ Loaded {len(train_conv)} examples from train_conv.pkl")

    # Print the 5th item (index 4) if it exists
    if len(train_conv) >= 5:
        print("🔍 Example #5:")
        print(train_conv[4])
    else:
        print("⚠️ Less than 5 examples in the file.")

except Exception as e:
    print(f"❌ Failed to load pickle file: {e}")
