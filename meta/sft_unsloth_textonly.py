from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import standardize_sharegpt
import wandb

# 0) W&B login
wandb.login()
wandb.init()


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
# 1) Load your JSONL as one big dataset
ds0 = load_dataset(
    "json",
    data_files="sft_data_textonly.jsonl",
    split="train",   # “train” here just means “don’t pre-split”—you’ll split yourself
)

# 2) Randomly split 10% off for validation
splits = ds0.train_test_split(test_size=0.1, shuffle=False, seed=42)

raw = splits["train"]
valid_ds = splits["test"]    # by convention “test” is the hold-out split


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
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4,
        max_steps     = 60,
        learning_rate = 1e-4,   # (or 2e-4 if you wish)
        bf16          = True,   # ✅ turn *on* bfloat-16
        fp16          = False,  # ❌ turn *off* fp16
        output_dir    = "outputs",
        seed          = 3407,
        logging_steps = 1,
        report_to="wandb", 
    ),
)


# from unsloth.chat_templates import train_on_responses_only
# trainer = train_on_responses_only(
#     trainer,
#     instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
#     response_part   ="<|start_header_id|>assistant<|end_header_id|>\n\n",
# )

wandb.watch(model, log="all", log_freq=50)
trainer_stats = trainer.train()
print(trainer_stats)

model.save_pretrained("Llama-3.2-3B-Instruct-bnb-4bit-idk")
tokenizer.save_pretrained("Llama-3.2-3B-Instruct-bnb-4bit-idk")

# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

#from unsloth import FastLanguageModel
def inference(model):
    if False:
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    # alpaca_prompt = You MUST copy from above!

    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "What is a famous tall tower in Paris?", # instruction
            "", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    # from transformers import TextStreamer
    # text_streamer = TextStreamer(tokenizer)
    # _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    tokenizer.batch_decode(outputs)

