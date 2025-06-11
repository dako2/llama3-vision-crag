# """
# Download huggingface data and load into json
# """
from datasets import load_dataset
import json

# Step 1: Load full dataset from Hugging Face (validation split)
dataset = load_dataset("crag-mm-2025/crag-mm-single-turn-public", split="validation")

# Step 2: Remove the "image" column
dataset = dataset.remove_columns(["image"])

# Step 3: Convert to a list of dicts
data_list = dataset.to_list()

# Step 4: Save to a JSON file
with open("crag-mm-validation-no-image.json", "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f"Saved {len(data_list)} examples to crag-mm-validation-no-image.json")

"""
Read the first row of data
"""
import json


# @E: session_id -> print("Answer:", entry.get("answers", {}).get("ans_full")[0])
# Step 1: Load the JSON file
with open("crag-mm-validation-no-image.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 2: Access the first item (index 0)
entry = data[0]

print("\nAll fields in entry:")
for k, v in entry.items():
    print(f"{k}: {v}")

session_id = entry.get("session_id", {})
query = entry.get("turns", {}).get("query")[0]
domain = entry.get("turns", {}).get("domain")[0]
query_category = entry.get("turns", {}).get("query_category")[0]
dynamism = entry.get("turns", {}).get("dynamism")[0]

# Step 3: Print key fields
print("=== Entry 10 ===")
print("session_id:", session_id)
print("Query:", query)
print("domain:", domain)
print("query_category:", query_category)
print("dynamism:", dynamism)
#print("Answer:", entry.get("answers", {}).get("ans_full")[0])


# """
# find the image based on session_id
# """

from datasets import load_dataset
dataset = load_dataset("crag-mm-2025/crag-mm-single-turn-public", split="validation")

def find_image_from_session_id(session_id):

    target_session_id = session_id
    # Find the matching example
    match = next((item for item in dataset if item["session_id"] == target_session_id), None)

    # Step 3: Load and display the image
    if match:
        image = match["image"]
        print("Session ID:", match["session_id"])
        print("Question:", match["turns"])
        print("Answer:", match["answers"]["ans_full"])
        print("PIL Image Object:", image)
        print("Image size:", image.size)
        print("Image mode:", image.mode)
    else:
        print("No matching session_id found.")

    
