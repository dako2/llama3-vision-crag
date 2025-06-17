#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import ast
import pandas as pd
import torch
import wandb

from datasets import Dataset
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model

# ------------------------------------------------------------------------------
# 1. Environment & logging
# ------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

wandb.login()
wandb.init(
    project="cragmm-huggingface-sft",
    name="llama3-vision-sft",
)

# ------------------------------------------------------------------------------
# 2. Model & PEFT configuration
# ------------------------------------------------------------------------------
ckpt = "meta-llama/Llama-3.2-11B-Vision"
USE_LORA = False
FREEZE_IMAGE = True
FREEZE_LLM = False  # must be False if FREEZE_IMAGE=True

if USE_LORA:
    # QLoRA + LoRA adapter
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = MllamaForConditionalGeneration.from_pretrained(
        ckpt,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","v_proj","k_proj","o_proj","down_proj","up_proj"],
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()

elif FREEZE_IMAGE:
    if FREEZE_LLM:
        raise ValueError("Cannot freeze both image encoder and text decoder.")
    model = MllamaForConditionalGeneration.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # freeze vision encoder
    for p in model.vision_model.parameters():
        p.requires_grad = False

elif FREEZE_LLM:
    model = MllamaForConditionalGeneration.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # freeze text decoder
    for p in model.language_model.parameters():
        p.requires_grad = False

else:
    model = MllamaForConditionalGeneration.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

processor = AutoProcessor.from_pretrained(ckpt)

# ------------------------------------------------------------------------------
# 3. Load & preprocess DataFrame
# ------------------------------------------------------------------------------
df = pd.read_csv("turn_evaluation_results_all_1p3k.csv")

def safe_parse_and_extract(msg):
    if pd.isna(msg):
        return ""
    try:
        parsed = ast.literal_eval(msg)
        return parsed[0]["content"][0]["text"]
    except Exception:
        return ""

def parse2(msg):
    if pd.isna(msg):
        return ""
    try:
        return ast.literal_eval(msg)[0]
    except Exception:
        return ""

# extract user_text
df["user_text"] = df["messages"].apply(safe_parse_and_extract)
df.loc[df["user_text"] == "", "user_text"] = df["query"]

# prepare finetune_answer
df["ground_truth"] = df["ground_truth"].apply(parse2)
df["finetune_answer"] = df["ground_truth"]
df.loc[df["api_response"] == "{'accuracy': True}", "finetune_answer"] = (
    "i don't know. it's difficult to answer the question accurately."
)
df.loc[df["is_miss"] == True, "finetune_answer"] = "i don't know"

# build Hugging Face dataset
ds = Dataset.from_dict({
    "user_text": df["user_text"].tolist(),
    "finetune_answer": df["finetune_answer"].tolist(),
})

# ------------------------------------------------------------------------------
# 4. Tokenization / Pre-tokenize
# ------------------------------------------------------------------------------
max_length = 512

def tokenize_fn(examples):
    prompts = [
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{a}<|eot_id|>"
        for u, a in zip(examples["user_text"], examples["finetune_answer"])
    ]
    model_inputs = processor(
        text=prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    # build labels: mask pad token
    labels = [
        [token if token != processor.tokenizer.pad_token_id else -100 for token in seq]
        for seq in input_ids
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# map and remove raw text columns
ds = ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=["user_text", "finetune_answer"],
)

# set to torch tensors
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ------------------------------------------------------------------------------
# 5. Training setup
# ------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=1e-6,
    warmup_steps=2,
    logging_steps=10,
    report_to="wandb",
    save_strategy="no",
    push_to_hub=True,
    bf16=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=default_data_collator,
)

wandb.watch(model, log="all", log_freq=50)

# ------------------------------------------------------------------------------
# 6. Run training
# ------------------------------------------------------------------------------
trainer.train()
