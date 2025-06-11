#sft data generation

from datasets import load_dataset
import json
from typing import Dict, List, Any, Optional
from PIL import Image

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
        return image
    else:
        print("No matching session_id found.")
        return None
    
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
ground_truth_answer = entry.get("answers", {}).get("ans_full")[0]
image = find_image_from_session_id(session_id)
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
print("Image size:", image.size)
print("Image mode:", image.mode)

from remote_search_pipeline import RemoteSearchPipeline
from agents.rag_agent import SimpleRAGAgent

# image_hits = search_pipeline(image, k=5)
# text_hits  = search_pipeline(query, k=5)

queries = [query]
images = [image]
message_histories = [None]


TARGET_WIDTH = 960
TARGET_HEIGHT = 1280
expected_size = (TARGET_WIDTH, TARGET_HEIGHT)
def resize_images(images: List[Image.Image], target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT) -> List[Image.Image]:
    resized_images = []
    for img in images:
        if img.size != (target_width, target_height):
            img = img.resize((target_width, target_height), Image.LANCZOS)
        resized_images.append(img)
    return resized_images

images = resize_images(images)
def main():
    search_pipeline = RemoteSearchPipeline("http://localhost:8001")
    agent = SimpleRAGAgent(search_pipeline)      # vLLM spawns here
    #answers = agent.batch_generate_response(queries, images, message_histories)
    answers = agent.batch_summarize_images(queries, images)

    # … rest of your driver code …

if __name__ == "__main__":
    main()