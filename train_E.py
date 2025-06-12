import wandb
import pickle
from sft_unsloth_E import load_model, run_training
# Step 1: Load or preprocess training data
# train_conv = load_sft_dataset("results/selected_pipeline_finetune_data_final.jsonl", "sft_dataset.pkl")

# Load the saved dataset
with open("results/train_conv.pkl", "rb") as f:
    train_conv = pickle.load(f)

# Step 2: Load model and tokenizer
model, tokenizer = load_model()

# Step 3: Start training
run_training(train_conv, model, tokenizer)