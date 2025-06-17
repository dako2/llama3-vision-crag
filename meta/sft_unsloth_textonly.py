from unsloth import FastLanguageModel, FastVisionModel
import torch
from datasets import load_dataset
import pandas as pd

from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import standardize_sharegpt
import wandb

from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


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

#instruction = "You are an expert radiographer. Describe accurately what you see in this image."
instruction = ""
def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : sample["user_text"].values},
            #{"type" : "image", "image" : sample["image"]} 
            ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["finetune_answer"].values} ]
        },
    ]
    return { "messages" : conversation }

df = pd.read_csv("turn_evaluation_results_all_1p3k.csv")

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

converted_dataset = [convert_to_conversation(sample) for sample in df]

print(converted_dataset[0])

if True:

    data_collator = UnslothVisionDataCollator(
        model,
        tokenizer,
        train_on_responses_only = False,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    FastVisionModel.for_training(model) # Enable for training!

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = data_collator, # Must use!
        train_dataset = converted_dataset,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 30,
            # num_train_epochs = 1, # Set this instead of max_steps for full training runs
            learning_rate = 2e-4,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",     # For Weights and Biases

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 2048,
        ),
    )
    
    trainer_stats = trainer.train()
