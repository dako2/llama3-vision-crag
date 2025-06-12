# finetune_runner.py

import os
import json
import pickle
import torch
import wandb
from tqdm.auto import tqdm
from transformers import TrainerCallback
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# wandb.login()
# wandb.init()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# def load_sft_dataset(jsonl_path="selected_pipeline_finetune_data_final.jsonl", save_as="sft_dataset.pkl"):
#     dataset = []
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for line in tqdm(f, desc="Loading JSONL"):
#             example = json.loads(line)

#             # ✅ Only include if accuracy is 1 or -1
#             if example.get("accuracy") not in [-1, 1]:
#                 continue

#             user_msg = None
#             for msg in example["messages"]:
#                 if msg["role"] == "user":
#                     user_msg = [{"type": "text", "text": msg["content"]}]
#                     break
#             if not user_msg:
#                 continue

#             messages = [
#                 {"role": "user", "content": user_msg},
#                 {"role": "assistant", "content": [{"type": "text", "text": example["finetune_output"]}, {"type": "image", "image": None}]}
#             ]
#             dataset.append({"messages": messages})

#     with open(save_as, "wb") as f:
#         pickle.dump(dataset, f)

#     print(f"✅ Saved {len(dataset)} examples with accuracy == 1 or -1 to {save_as}")
#     return dataset

def load_model():
    model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"
    model, tokenizer, processor = FastVisionModel.from_pretrained(
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
    return model, tokenizer, processor

class GPUStats(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            for i in range(torch.cuda.device_count()):
                wandb.log({
                    f"gpu{i}_alloc": torch.cuda.memory_allocated(i)/1e9,
                    f"gpu{i}_reserved": torch.cuda.memory_reserved(i)/1e9,
                }, step=state.global_step)

def run_training(train_data, model, tokenizer):
    config = SFTConfig(
        per_device_train_batch_size=32,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        optim="adamw_8bit",
        bf16=is_bf16_supported(),
        fp16=False,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",
        run_name="cragmm-vision-lora2-E",
        logging_steps=10,
        dataset_text_field="messages",
        # dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        max_seq_length=8192,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_data,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        callbacks=[GPUStats()],
    )

    wandb.watch(model, log="all", log_freq=50)
    trainer.train()

    model.save_pretrained("llama3-vision-finetuned_E")
    tokenizer.save_pretrained("llama3-vision-finetuned_E")
