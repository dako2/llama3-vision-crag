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
 
agent = SimpleRAGAgent(search_pipeline)      # vLLM spawns here

from datasets import load_dataset
from itertools import islice

############################
# Config
############################
BATCH_SIZE = 1                    # ← change to fit GPU/VRAM
SPLIT      = "validation"         # or "public_test", etc.

############################
# 1.  Load the HF split
############################
ds = load_dataset(
    "crag-mm-2025/crag-mm-single-turn-public",
    split=SPLIT,
    streaming=False,              # set True if RAM is tight
)

############################
# 2.  Mini helper for batching
############################
def batched(iterable, n):
    "Yield successive n-sized lists."
    it = iter(iterable)
    while (batch := list(islice(it, n))):
        yield batch

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

    
############################
# 3.  Main loop
############################
for minibatch in batched(ds, BATCH_SIZE):
    # Each row has: session_id, image (PIL), turns[...] … :contentReference[oaicite:0]{index=0}
    session_ids = [row["session_id"] for row in minibatch]
    queries   = [row["turns"]["query"][0] for row in minibatch]
    images_raw = [row["image"]             for row in minibatch]   # PIL.Image
    images    = resize_images(images_raw)
    histories = [None] * len(minibatch)                          # no message history

    # ── retrieval (optional) ────────────────────────────────

    #answers = agent.batch_generate_response(queries, images, message_histories)
    answers = agent.batch_generate_response(session_ids, queries, images, message_histories)
     