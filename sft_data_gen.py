#sft data generation

from datasets import load_dataset
import json
from typing import Dict, List, Any, Optional
from PIL import Image
 
from collections import defaultdict
from itertools import islice

from remote_search_pipeline import RemoteSearchPipeline
from agents.rag_agent import SimpleRAGAgent
from cragmm_search.search import UnifiedSearchPipeline

search_pipeline = UnifiedSearchPipeline(
    text_model_name="BAAI/bge-large-en-v1.5",
    image_model_name="openai/clip-vit-large-patch14-336",
    web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
)
############################
# 1.  Up-front data loading
############################
dataset = load_dataset(
    "crag-mm-2025/crag-mm-single-turn-public",
    split="validation",
    streaming=False,   # keeps things in RAM for fast look-ups
)

# Build an O(1) lookup table: session_id → dataset row
id2row = {row["session_id"]: row for row in dataset}

with open("crag-mm-validation-no-image.json", "r", encoding="utf-8") as f:
    raw_records = json.load(f)

##################################################
# 2.  Helper: yield successive fixed-size batches
##################################################
def batched(iterable, n):
    "s -> (s0…s{n-1}), (sn…s{2n-1}), …"
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

###############################################
# 3.  Convert each record into model inputs
###############################################
def to_model_inputs(record):
    """Return (query, image, message_history) or None if the image is missing."""
    row = id2row.get(record["session_id"])
    if row is None or row["image"] is None:
        return None                          # skip if we can’t find the picture

    query = record["turns"]["query"][0]
    image = row["image"]                    # PIL.Image object
    msg_history = None                      # or previous turns if you have them
    return query, image, msg_history

model_inputs = [x for x in map(to_model_inputs, raw_records) if x]


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

def main():
    #search_pipeline = RemoteSearchPipeline("http://localhost:8001")

    agent = SimpleRAGAgent(search_pipeline)      # vLLM spawns here
    #answers = agent.batch_generate_response(queries, images, message_histories)
    #answers = agent.batch_summarize_images(queries, images)
    
    #################################################
    # 4.  Main loop – send each batch to your agent
    #################################################
    BATCH_SIZE = 8
    all_agent_outputs = []

    for batch in batched(model_inputs, BATCH_SIZE):
        images = resize_images(images)
        queries, images, histories = zip(*batch)          # unzip
        # RemoteSearchPipeline can work in parallel here too if you like:
        agent_responses, msg_states = agent.batch_generate_response(
            list(queries), list(images), list(histories)
        )
        all_agent_outputs.extend(agent_responses)

    ############################################################
    # 5. (Optional) emit SFT lines in Alpaca / ChatML style JSON
    ############################################################
    with open("sft_batch_results.jsonl", "w", encoding="utf-8") as out:
        for record, answer in zip(raw_records, all_agent_outputs):
            out.write(json.dumps({
                "session_id": record["session_id"],
                "prompt":     record["turns"]["query"][0],
                "response":   answer,
            }) + "\n")


# # Step 2: Access the first item (index 0)
# entry = data[0]

# print("\nAll fields in entry:")
# for k, v in entry.items():
#     print(f"{k}: {v}")

# session_id = entry.get("session_id", {})
# query = entry.get("turns", {}).get("query")[0]
# ground_truth_answer = entry.get("answers", {}).get("ans_full")[0]
# image = find_image_from_session_id(session_id)
# domain = entry.get("turns", {}).get("domain")[0]
# query_category = entry.get("turns", {}).get("query_category")[0]
# dynamism = entry.get("turns", {}).get("dynamism")[0]

# # Step 3: Print key fields
# print("=== Entry 10 ===")
# print("session_id:", session_id)
# print("Query:", query)
# print("domain:", domain)
# print("query_category:", query_category)
# print("dynamism:", dynamism)
# print("Image size:", image.size)
# print("Image mode:", image.mode)

# image_hits = search_pipeline(image, k=5)
# text_hits  = search_pipeline(query, k=5)


if __name__ == "__main__":
    main()