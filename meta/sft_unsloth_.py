#!/usr/bin/env python
# finetune_llama3_vision.py
# ---------------------------------------------------------------
# Fine-tune Llama-3.2-11B-Vision with Unsloth + new SFT JSONL
# ---------------------------------------------------------------
"""
python finetune_llama3_vision.py  \
  --jsonl sft_data.jsonl          \
  --split validation              # or train/test, etc.

"""
import os, pickle, json, argparse
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Optional

import torch
import wandb
from tqdm.auto import tqdm
from PIL import Image
from datasets import load_dataset, Image as HFImage

from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--jsonl", default="sft_data.jsonl",
                    help="SFT JSONL file produced by postprocess_sft_labeling.py")
parser.add_argument("--split", default="public_test",
                    help="Which CRAG-MM split to pull images from")
parser.add_argument("--target_w", type=int, default=960)
parser.add_argument("--target_h", type=int, default=1280)
parser.add_argument("--cache", default="train_conv.pkl",
                    help="Pickle cache of processed conversations")
args = parser.parse_args()

JSONL_PATH  = args.jsonl
PICKLE_PATH = args.cache
TARGET_SIZE = (args.target_w, args.target_h)

# ----------------------------------------------------------------
# 0)  Weights & Biases
# ----------------------------------------------------------------
wandb.login()
wandb.init(project="cragmm-lora", name="llama3-vision-sft")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ----------------------------------------------------------------
# 1)  Build image lookup table
# ----------------------------------------------------------------
print(f"📥 Loading CRAG-MM “{args.split}” split for images…")
crag_ds = load_dataset("crag-mm-2025/crag-mm-single-turn-public",
                       split=args.split)
crag_ds = crag_ds.cast_column("image", HFImage(decode=True))
IMAGE_MAP = {row["session_id"]: row["image"] for row in crag_ds}

# ----------------------------------------------------------------
# 2)  Helpers
# ----------------------------------------------------------------
def _resize(img: Image.Image) -> Image.Image:
    return img.resize(TARGET_SIZE, Image.LANCZOS) if img.size != TARGET_SIZE else img


def _inject_real_image(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    1. Verify the first user turn has *exactly* 2 items: [text, image-placeholder]
    2. Swap the placeholder string with the real PIL image.
    3. Remove any keys whose value is None:
         • from the text part  → drop `"image"` if it’s null
         • from the image part → drop `"text"` (or any other null key)
    4. If the picture is missing, drop the entire conversation.
    """
    try:
        sid = rec.get("session_id")

        # ── locate first user message ─────────────────────────────────────────
        user_msg = next((m for m in rec["messages"] if m["role"] == "user"), None)
        if user_msg is None or len(user_msg["content"]) != 2:
            return None

        text_part, img_part = user_msg["content"]

        # ── text block must be type=text ──────────────────────────────────────
        if text_part.get("type") != "text":
            return None
        text_part.pop("image", None)                    # ← drop stray key

        # ── image block must be type=image ────────────────────────────────────
        if img_part.get("type") != "image":
            return None

        img_key = img_part.get("image") or sid          # placeholder string
        img = IMAGE_MAP.get(img_key)
        if img is None:
            return None                                 # picture missing → skip

        img_part["image"] = _resize(img)                # ✅ real PIL image
        img_part.pop("text", None)                      # ← drop `"text": None`
        # (If other spurious keys appear, pop them the same way.)

        return rec

    except Exception as e:
        print(f"[ERROR] {sid} – {e}")
        return None

def load_or_build_dataset() -> List[Dict[str, Any]]:
    if os.path.exists(PICKLE_PATH):
        print(f"📦  Using cached dataset → {PICKLE_PATH}")
        with open(PICKLE_PATH, "rb") as f:
            return pickle.load(f)

    print("⚙️   Building dataset (multiprocessing)…")
    rows = load_dataset("json", data_files=JSONL_PATH, split="train")

    with Pool(cpu_count()) as pool:
        processed = list(
            tqdm(pool.imap_unordered(_inject_real_image, rows),
                 total=len(rows),
                 desc="Inject images")
        )
    processed = [x for x in processed if x]
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(processed, f)
    print(f"✅  Cached dataset → {PICKLE_PATH}")
    return processed

train_conv = load_or_build_dataset()
assert all(
    m["content"][0]["type"] == "text"  and "image" not in m["content"][0]
    and
    m["content"][1]["type"] == "image" and isinstance(m["content"][1]["image"], Image.Image)
    for row in train_conv
    for m in row["messages"] if m["role"] == "user"
)
print("Total conversations:", len(train_conv))

# ----------------------------------------------------------------
# 3)  Model & LoRA prep
# ----------------------------------------------------------------
model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"
model, tokenizer = FastVisionModel.from_pretrained(
    model_id,
    load_in_4bit=False,
    use_gradient_checkpointing="unsloth",
)
FastVisionModel.for_training(model)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0.0,
    bias="none", random_state=3443,
)

# ----------------------------------------------------------------
# 4)  GPU memory callback
# ----------------------------------------------------------------
class GPUStats(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            for i in range(torch.cuda.device_count()):
                wandb.log({
                    f"gpu{i}_alloc": torch.cuda.memory_allocated(i) / 1e9,
                    f"gpu{i}_reserved": torch.cuda.memory_reserved(i) / 1e9,
                }, step=state.global_step)

# ----------------------------------------------------------------
# 5)  SFT Trainer config
# ----------------------------------------------------------------
config = SFTConfig(
    per_device_train_batch_size=32,          # adjust for VRAM
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    optim="adamw_8bit",
    bf16=True, fp16=False,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="wandb",
    run_name="cragmm-vision-lora",
    logging_steps=10,
    dataset_text_field="messages",           # we pass list-of-dicts
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    max_seq_length=8192,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=config,
    train_dataset=train_conv,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    callbacks=[GPUStats()],
)

# ----------------------------------------------------------------
# 6)  Train & save
# ----------------------------------------------------------------
wandb.watch(model, log="all", log_freq=50)
trainer.train()

os.makedirs("llama3-vision-finetuned", exist_ok=True)
model.save_pretrained("llama3-vision-finetuned")
tokenizer.save_pretrained("llama3-vision-finetuned")
print("🎉  Fine-tuning complete – model saved to ./llama3-vision-finetuned")
