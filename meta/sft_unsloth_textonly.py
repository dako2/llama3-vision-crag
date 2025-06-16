from unsloth import FastLanguageModel
import torch
from datasets import load_dataset

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    #"unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    #"unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    #"unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

]
# … imports and FP16/4-bit flags …

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=3407,
)

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")

# --------------  LOAD + CONVERT DATA  --------------
raw = load_dataset("json", data_files="sft_data_textonly.jsonl", split="train")

def strip_images_and_flatten(example):
    conv = []
    for msg in example["messages"]:
        texts = [c["text"] for c in msg["content"] if c["type"]=="text"]
        if texts:
            conv.append({"role": msg["role"], "content": "\n".join(texts).strip()})
    return {"conversations": conv}

ds = raw.map(strip_images_and_flatten, remove_columns=["messages"])

def to_chat_template(examples):
    return {"text": [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in examples["conversations"]
    ]}

ds = ds.map(to_chat_template, batched=True)
assert all(len(x) > 0 for x in ds["text"])

# --------------  TRAIN  --------------
trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = ds,
    dataset_text_field = "text",
    max_seq_length     = 2048,
    data_collator      = DataCollatorForSeq2Seq(tokenizer),
    packing            = False,
    args = TrainingArguments(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        max_steps=60, learning_rate=2e-4, fp16=True,
        output_dir="outputs", seed=3407, logging_steps=1, report_to="none",
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part   ="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

trainer_stats = trainer.train()
print(trainer_stats)
