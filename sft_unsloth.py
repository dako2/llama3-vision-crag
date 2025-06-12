import torch
import wandb
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

# 0) W&B login
wandb.login()
wandb.init()

# 1) Load dataset created by your script
# This file already has the correct chat format in the "messages" column.
train_conv = load_dataset("json", data_files="sft_data.jsonl", split="train")

# 2) Add a placeholder for the image to satisfy the Vision model's data requirements.
# This prevents the "Invalid input type" ValueError.
def add_empty_image(example):
    example["image"] = None
    return example

train_conv = train_conv.map(add_empty_image)

print(f"Dataset loaded and prepared. First example:\n{train_conv[0]}")

# 3) Load & prepare model
model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"
model, tokenizer = FastVisionModel.from_pretrained(
    model_id,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)
FastVisionModel.for_training(model)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    random_state=3443,
    use_rslora=False,
    loftq_config=None,
)

# 4) GPU logging callback
class GPUStats(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            for i in range(torch.cuda.device_count()):
                wandb.log({
                  f"gpu{i}_alloc": torch.cuda.memory_allocated(i)/1e9,
                  f"gpu{i}_reserved": torch.cuda.memory_reserved(i)/1e9,
                }, step=state.global_step)

# 5) Configure & run SFTTrainer
training_args = SFTConfig(
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    optim="adamw_8bit",
    bf16=True,
    fp16=False,
    save_strategy="epoch",
    save_total_limit=None,
    report_to="wandb",
    run_name="llama3-vision-text-finetune",
    logging_steps=10,
    max_seq_length=8192,
    # DO NOT set dataset_text_field.
    # By leaving it unset, SFTTrainer will automatically apply the
    # tokenizer's chat template to the "messages" column.
    # This prevents the "'dict' object has no attribute 'startswith'" AttributeError.
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_conv,
    # Use the UnslothVisionDataCollator to correctly handle text and image (None) inputs
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    callbacks=[GPUStats()],
)

wandb.watch(model, log="all", log_freq=50)
print("Starting training...")
trainer.train()
print("Training finished.")

# 6) Save the final model
output_dir = "llama3-vision-finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model saved to {output_dir}")