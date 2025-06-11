from multiprocessing import Pool
from tqdm.auto import tqdm
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import pandas as pd
import torch
import wandb
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import pickle
import os
# 0) W&B login
wandb.login()
wandb.init()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_sft_dataset(pickle_path="sft_dataset.pkl"):
    """Load and return the SFT dataset from a pickle file."""
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Loaded {len(data)} examples from {pickle_path}")
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {pickle_path}")
        return []
    except Exception as e:
        print(f"❌ Failed to load pickle file: {e}")
        return []

train_conv = load_sft_dataset("sft_dataset.pkl")
print(train_conv[-1])
print(len(train_conv))

# # 5) Load & prepare model
model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"
model, tokenizer = FastVisionModel.from_pretrained(model_id, load_in_4bit=True, use_gradient_checkpointing="unsloth")
FastVisionModel.for_training(model)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0.0,
    bias="none", random_state=3443,
    use_rslora=False, loftq_config=None,
)

# 6) GPU logging callback
class GPUStats(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            for i in range(torch.cuda.device_count()):
                wandb.log({
                  f"gpu{i}_alloc": torch.cuda.memory_allocated(i)/1e9,
                  f"gpu{i}_reserved": torch.cuda.memory_reserved(i)/1e9,
                }, step=state.global_step)

# 7) Configure & run SFTTrainer
config = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    optim="adamw_8bit",
    bf16=is_bf16_supported(), fp16=False,
    save_strategy="epoch", save_total_limit=1,
    report_to="wandb", run_name="cragmm-vision-lora2", logging_steps=10,
    dataset_text_field="messages",
    dataset_kwargs={"skip_prepare_dataset":True},
    remove_unused_columns=False,
    max_seq_length=8192,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=config,
    train_dataset=train_conv,                     # <-- list of dicts, no None
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    callbacks=[GPUStats()],
)

wandb.watch(model, log="all", log_freq=50)
trainer.train()
model.save_pretrained("llama3-vision-finetuned")
tokenizer.save_pretrained("llama3-vision-finetuned")


# try:
#     with open("train_conv.pkl", "rb") as f:
#         train_conv = pickle.load(f)
# except:
#     # 1) GPU info
#     print("CUDA:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())

#     # 2) Load correct IDs
#     df = pd.read_csv("correct_answer_index.csv")
#     correct_set = set(df["session_id"])

#     SYSTEM_PROMPT = (
#         "You are a helpful assistant that truthfully answers user questions about the provided image. "
#         "Provide the answer in one sentence. If you don't know the answer, respond with 'I don't know'. Here is the question: "
#     )
#     from PIL import UnidentifiedImageError

#     def preprocess_to_conv(ex):
#         # 1) rewrite
        
#         #if ex["session_id"] not in correct_set:
#         #    ex["answers"] = [{"ans_full": "I don't know"}]

#         # 2) pick image or URL
#         img = ex.get("image") or ex.get("image_url")
        
#         try:
#             if isinstance(img, bytes):
#                 img = Image.open(BytesIO(img)).convert("RGB")
#             elif isinstance(img, str) and img.startswith(("http://", "https://")):
#                 resp = requests.get(img, timeout=5)
#                 resp.raise_for_status()
#                 img = Image.open(BytesIO(resp.content)).convert("RGB")
#             # else assume it's already a PIL.Image

#             # 4) resize if needed (ensure max 1120x1120, keep aspect ratio)
#             max_side = 1120
#             if img.width > max_side or img.height > max_side:
#                 img.thumbnail((max_side, max_side), Image.LANCZOS)
#             # (img.thumbnail resizes in-place and keeps aspect ratio)

#         except (UnidentifiedImageError, requests.RequestException, OSError) as e:
#             # skip broken URLs or unreadable bytes
#             return None

#         # 4) build messages
#         instr = ex["turns"][0]["query"]
#         rsp   = ex["answers"][0]["ans_full"]
#         return {
#             "messages": [
#                 {"role":"user", "content":[{"type":"image","image":img},
#                                         {"type":"text","text":SYSTEM_PROMPT+instr}]},
#                 {"role":"assistant","content":[{"type":"text","text":rsp}]},
#             ]
#         }

#     # 4) Parallel preprocess + filter
#     raw   = load_dataset("crag-mm-2025/crag-mm-single-turn-public", revision="v0.1.1")["validation"]
#     with Pool(16) as p:
#         all_conv = list(tqdm(
#             p.imap(preprocess_to_conv, raw),
#             total=len(raw),
#             desc="Preprocessing"
#         ))

#     train_conv = [c for c in all_conv if c is not None]
#     print(f"✅ Kept {len(train_conv)}/{len(all_conv)} examples after dropping bad images")


#     with open("train_conv.pkl", "wb") as f:
#         pickle.dump(train_conv, f)
 
# load all_llm_calls.jsonl into train_conv for sft
