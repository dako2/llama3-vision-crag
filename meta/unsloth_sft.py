from unsloth import FastLanguageModel, FastVisionModel
import torch
from datasets import load_dataset, Dataset
import pandas as pd
import wandb

from transformers import TrainingArguments, DataCollatorForSeq2Seq, AutoProcessor

from trl import SFTTrainer, SFTConfig
import ast 
from datasets import Dataset

def safe_parse_and_extract(msg):
    if pd.isna(msg):
        return ""

    parsed = ast.literal_eval(msg)
    return parsed[0]["content"][0]["text"]

def parse2(msg):
    if pd.isna(msg):
        return ""

    parsed = ast.literal_eval(msg)
    return parsed[0]

# 0) W&B login
wandb.login()
wandb.init()

max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.bfloat16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-11B-Vision",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

#load the dataset
# Step 1: Load the dataset
df = pd.read_csv("turn_evaluation_results_all_1p3k.csv")
df["user_text"] = df["messages"].apply(safe_parse_and_extract)
df.loc[df["user_text"] == "", "user_text"] = df["query"]
# Step 2: Clean finetune_answer for specific API response
df["ground_truth"] = df["ground_truth"].apply(parse2)
df["finetune_answer"] = df["ground_truth"]  # Ensure column exists
df.loc[df["api_response"] == "{'accuracy': True}", "finetune_answer"] = "i don't know. there is not enough information to answer the question accurately."
df.loc[df["is_miss"] == True, "finetune_answer"] = "i don't know"

# Step 3: Convert to HF Datase
ds = Dataset.from_dict({
    'finetune_answer': df['finetune_answer'].tolist(),
    'user_text': df['user_text'].tolist(),
})


def collate_fn(examples):
    texts = [
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        for user, answer in zip(examples["user_text"], examples["finetune_answer"])
    ]
    #print(texts)
    batch = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    print(batch)
    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  #
    # Ignore the image token index in the loss computation (model specific)
    #image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    #labels[labels == image_token_id] = -100
    
    batch["labels"] = labels

    return batch

ds = ds.map(collate_fn, batched=True)
ds = ds.remove_columns(["finetune_answer", "user_text"])

# data_collator = UnslothVisionDataCollator(
#     model,
#     processor.tokenizer,
#     train_on_responses_only = True,
#     instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
#     response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",        
# )

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,

    data_collator = DataCollatorForSeq2Seq(tokenizer=processor), # Must use!
    train_dataset = ds,
    
    args = SFTConfig(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        warmup_steps = 0,
        max_steps = 10,
        num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 1e-5,
        fp16 = False,
        bf16 = True,
        logging_steps = 10, 
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        save_strategy="epoch",
        save_total_limit=3,
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = max_seq_length,
    ),
)

wandb.watch(model, log="all", log_freq=50)
trainer_stats = trainer.train()
