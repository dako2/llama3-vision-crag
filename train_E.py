import wandb
from sft_unsloth_E import load_sft_dataset

# Step 1: Load or preprocess training data
train_conv = load_sft_dataset("results/selected_pipeline_finetune_data_final.jsonl", "sft_dataset.pkl")

import pickle
from sft_unsloth_E import load_model, run_training

# Load the saved dataset
with open("sft_dataset.pkl", "rb") as f:
    train_conv = pickle.load(f)

# Print the 5th example
print(train_conv[4])

from sft_unsloth import load_sft_dataset, load_model, run_training
# Step 2: Load model and tokenizer
model, tokenizer = load_model()

# Step 3: Start training
run_training(train_conv, model, tokenizer)