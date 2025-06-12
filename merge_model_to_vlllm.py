from unsloth import FastVisionModel
from transformers import AutoTokenizer

# Load base model and fine-tuned adapter
model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"
model, tokenizer = FastVisionModel.from_pretrained(model_id, load_in_4bit=False)
model.load_adapter("llama3-vision-finetuned_full")

# Merge and save in 16-bit HuggingFace format (for vLLM or transformers)
model.save_pretrained_merged("llama3-vision-merged-vllm", tokenizer, save_method="merged_16bit")
