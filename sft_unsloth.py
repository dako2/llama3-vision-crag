from multiprocessing import Pool
from tqdm.auto import tqdm
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import pandas as pd
import torch
import wandb
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import pickle
import os
from datasets import load_dataset
dataset = load_dataset("crag-mm-2025/crag-mm-single-turn-public", split="validation")

# 0) W&B login
wandb.login()
wandb.init()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from typing import List, Dict, Any
from tqdm.auto import tqdm
from datasets import load_dataset
from PIL import Image
from multiprocessing import Pool, cpu_count

# Preload CRAG-MM image dataset and build a lookup map
from datasets import Image as HFImage
crag_ds = load_dataset("crag-mm-2025/crag-mm-single-turn-public", split="validation")
crag_ds = crag_ds.cast_column("image", HFImage(decode=True))  # ✅ decode PIL images

IMAGE_MAP = {ex["session_id"]: ex["image"] for ex in crag_ds}

# Constants
TARGET_W, TARGET_H = 960, 1280

def resize_image(img: Image.Image) -> Image.Image:
    if img.size != (TARGET_W, TARGET_H):
        return img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    return img

# This function must be top-level for pickling in multiprocessing
def process_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    
    try:
        session_id = row["session_id"]
        img = IMAGE_MAP.get(session_id)
        if img is None:
            return None
        img = resize_image(img)

        messages = row["messages"]
        for turn in messages:
            if turn["role"] == "user":
                turn["content"].append({"type": "image", "image": img})
                break
        return {"messages": messages}
    except Exception as e:
        print(f"[ERROR] Failed processing session_id={row.get('session_id')}: {e}")
        return None

def load_sft_dataset_mp(jsonl_path: str = "sft_data.jsonl", num_workers: int = cpu_count()) -> List[Dict[str, Any]]:
    rows = load_dataset("json", data_files=jsonl_path, split="train")

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_row, rows), total=len(rows), desc="Building dataset"))

    results = [r for r in results if r is not None]
    return results

def load_or_build_dataset(pickle_path="train_conv.pkl", jsonl_path="sft_data.jsonl"):
    if os.path.exists(pickle_path):
        print(f"📦 Loading dataset from {pickle_path}")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    print("⚙️ Building dataset from JSONL with multiprocessing...")
    train_conv = load_sft_dataset_mp(jsonl_path)

    with open(pickle_path, "wb") as f:
        pickle.dump(train_conv, f)
    print(f"✅ Saved dataset to {pickle_path}")
    return train_conv

train_conv = load_or_build_dataset()


print("Total examples:", len(train_conv))

# # 5) Load & prepare model
model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"
model, tokenizer = FastVisionModel.from_pretrained(model_id, load_in_4bit=False, use_gradient_checkpointing="unsloth")
FastVisionModel.for_training(model)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0.0,
    bias="none", random_state=3443,
    use_rslora=False, loftq_config=None,
)

# 6) GPU logging callback
class GPUStats(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            for i in range(torch.cuda.device_count()):
                wandb.log({
                  f"gpu{i}_alloc": torch.cuda.memory_allocated(i)/1e9,
                  f"gpu{i}_reserved": torch.cuda.memory_reserved(i)/1e9,
                }, step=state.global_step)

# 7) Configure & run SFTTrainer
config = SFTConfig(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    optim="adamw_torch",                # <- full precision optimizer
    bf16=False,                         # <- force full precision
    fp16=False,
    save_strategy="epoch", 
    save_total_limit=1,
    report_to="wandb", 
    run_name="cragmm-vision-16bit",
    logging_steps=4,
    logging_dir="./logs",
    dataset_text_field="messages",
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    max_seq_length=8192,
)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=config,
    train_dataset=train_conv,                     # <-- list of dicts, no None
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    callbacks=[GPUStats()],
)

wandb.watch(model, log="all", log_freq=50)
trainer.train()
model.save_pretrained("llama3-vision-finetuned_full")
tokenizer.save_pretrained("llama3-vision-finetuned_full")

