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

TARGET_WIDTH  = 960
TARGET_HEIGHT = 1280
expected_size = (TARGET_WIDTH, TARGET_HEIGHT)

# ------------------------------------------------------------------------------
dataset = load_dataset("crag-mm-2025/crag-mm-single-turn-public", split="validation")

def find_image_from_session_id(session_id):
    target_session_id = session_id
    # Find the matching example
    match = next((item for item in dataset if item["session_id"] == target_session_id), None)

    # Step 3: Load and display the image
    if match:
        image = match["image"]
        return image
    else:
        return None

def resize_image(img: Image.Image,
                 w: int = TARGET_WIDTH,
                 h: int = TARGET_HEIGHT) -> Image.Image:
    """Resize PIL image if needed, keeping aspect ratio."""
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    return img

def load_sft_dataset(
    jsonl_path: str = "sft_data.jsonl",
) -> List[Dict[str, Any]]:
    """
    Build a training list compatible with Unslo​th Vision.
    Each element is   {"messages": [...]}
    """

    rows = load_dataset("json", data_files=jsonl_path, split="train")

    train_conv: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="Preparing training examples"):
        img = find_image_from_session_id(row["session_id"])
        img = resize_image(img)

        messages = row["messages"]

        # Insert the image into the user's content
        for turn in messages:
            if turn["role"] == "user":
                turn["content"].append({"type": "image", "image": img})
                break  # only insert once into first user turn

        train_conv.append({ "messages": messages })


    import pickle

    with open("train_conv.pkl", "wb") as f:
        pickle.dump(train_conv, f)


    return train_conv

try:
    with open("train_conv.pkl", "rb") as f:
        train_conv = pickle.load(f)
except:
    train_conv = load_sft_dataset("sft_data.jsonl")

print("Total examples:", len(train_conv))

# # 5) Load & prepare model
model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"
model, tokenizer = FastVisionModel.from_pretrained(model_id, load_in_4bit=True, use_gradient_checkpointing="unsloth")
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
    per_device_train_batch_size=32, # 32-48
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4, # 1e-4, 5e-5, 2e-4
    optim="adamw_8bit",
    bf16=is_bf16_supported(), fp16=False,
    save_strategy="epoch", save_total_limit=1,
    report_to="wandb", run_name="cragmm-vision-lora2", logging_steps=10,
    dataset_text_field="messages",
    dataset_kwargs={"skip_prepare_dataset":True},
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
model.save_pretrained("llama3-vision-finetuned")
tokenizer.save_pretrained("llama3-vision-finetuned")

