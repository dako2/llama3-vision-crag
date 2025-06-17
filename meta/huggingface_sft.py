#https://github.com/huggingface/huggingface-llama-recipes/blob/main/fine_tune/Llama-Vision%20FT.ipynb
#torch tune: https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_single_device.py
#llama-cookbook: https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/finetuning

from datasets import load_dataset
from datasets import Dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import ast
import pandas as pd

from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
import wandb
wandb.login()
wandb.init(project="cragmm-huggingface-sft", name="llama3-vision-sft")


ckpt = "meta-llama/Llama-3.2-11B-Vision"
USE_LORA = False
FREEZE_LLM = False
FREEZE_IMAGE = True

if USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        use_dora=True, # optional DoRA 
        init_lora_weights="gaussian"
    )

    model = MllamaForConditionalGeneration.from_pretrained(
            ckpt,
            torch_dtype=torch.bfloat16,
            device_map="auto"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

elif FREEZE_IMAGE:
    if FREEZE_LLM:
        raise ValueError("You cannot freeze image encoder and text decoder at the same time.")
    model = MllamaForConditionalGeneration.from_pretrained(ckpt,
        torch_dtype=torch.bfloat16, device_map="auto")
    # freeze vision model to save up on compute
    for param in model.vision_model.parameters():
        param.requires_grad = False

elif FREEZE_LLM:
    if FREEZE_IMAGE:
        raise ValueError("You cannot freeze image encoder and text decoder at the same time.")
    model = MllamaForConditionalGeneration.from_pretrained(ckpt,
        torch_dtype=torch.bfloat16, device_map="auto")
    # freeze text model, this is encouraged in paper
    for param in model.language_model.parameters():
        param.requires_grad = False
        
else: # full ft
    model = MllamaForConditionalGeneration.from_pretrained(ckpt,
        torch_dtype=torch.bfloat16, device_map="auto")

processor = AutoProcessor.from_pretrained(ckpt)


df = pd.read_csv("turn_evaluation_results_all_1p3k.csv")

def safe_parse_and_extract(msg):
    if pd.isna(msg):
        return ""
    try:
        parsed = ast.literal_eval(msg)
        return parsed[0]["content"][0]["text"]
    except Exception as e:
        return f"[PARSE_ERROR: {str(e)}]"

def parse2(msg):
    if pd.isna(msg):
        return ""
    try:
        parsed = ast.literal_eval(msg)
        return parsed[0]
    except Exception as e:
        return f"[PARSE_ERROR: {str(e)}]"


df["user_text"] = df["messages"].apply(safe_parse_and_extract)

df.loc[df["user_text"] == "", "user_text"] = df["query"]


# Step 2: Clean finetune_answer for specific API response
df["ground_truth"] = df["ground_truth"].apply(parse2)
df["finetune_answer"] = df["ground_truth"]  # Ensure column exists
df.loc[df["api_response"] == "{'accuracy': True}", "finetune_answer"] = "i don't know. it's difficult to answer the question accurately."
df.loc[df["is_miss"] == True, "finetune_answer"] = "i don't know"

# Step 3: Convert to Hugging Face dataset
ds = Dataset.from_dict({
    'finetune_answer': df['finetune_answer'].tolist(),
    'user_text': df['user_text'].tolist(),
})

# Step 4: Define process function
def process(examples):
    texts = [
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        for user, answer in zip(examples["user_text"], examples["finetune_answer"])
    ]

    batch = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == 128256] = -100
    batch["labels"] = labels

    # ✅ Move everything to GPU, but only cast float tensors to bfloat16
    for k in batch:
        if batch[k].dtype in [torch.float32, torch.float16, torch.bfloat16]:
            batch[k] = batch[k].to(torch.bfloat16)
        batch[k] = batch[k].to("cuda")

    return batch


ds = ds.map(process, batched=True)

from torch.utils.data import default_collate

def custom_collator(examples):
    # Assumes each example is pre-tokenized by `process`
    # If not pre-tokenized, move tokenization here instead
    batch = processor(text=[
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{ex['user_text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{ex['finetune_answer']}<|eot_id|>"
        for ex in examples
    ], return_tensors="pt", padding=True, truncation=True)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == 128256] = -100  # Optional image token ID masking
    batch["labels"] = labels


    # ✅ Move everything to GPU, but only cast float tensors to bfloat16
    for k in batch:
        if batch[k].dtype in [torch.float32, torch.float16, torch.bfloat16]:
            batch[k] = batch[k].to(torch.bfloat16)
        batch[k] = batch[k].to("cuda")

    return batch


from transformers import TrainingArguments
args=TrainingArguments(
            num_train_epochs=3,
            remove_unused_columns=False,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            report_to="wandb",
            logging_steps=10,
            save_strategy="no",
            optim="adamw_torch",
            push_to_hub=True,
            save_total_limit=1,
            bf16=True,
            output_dir="./lora",
            dataloader_pin_memory=False,
        )

from transformers import Trainer
trainer = Trainer(
    model=model,
    train_dataset=ds,
    data_collator=custom_collator,
    args=args
)
wandb.watch(model, log="all", log_freq=50)
trainer.train()
