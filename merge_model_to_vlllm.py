# from unsloth import FastVisionModel
# from transformers import AutoTokenizer

# # Load base model and fine-tuned adapter
# model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"
# model, tokenizer = FastVisionModel.from_pretrained(model_id, load_in_4bit=False)
# model.load_adapter("llama3-vision-finetuned_full")

# # Merge and save in 16-bit HuggingFace format (for vLLM or transformers)
# model.save_pretrained_merged("llama3-vision-merged-vllm", tokenizer, save_method="merged_16bit")


from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "trainer_output/checkpoint-12",
    max_seq_length = 8192,
    #dtype = dtype,
    load_in_4bit = True, # still 4-bits as trained model
)

# Saving & Pushing
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
#model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# model.save_pretrained(
#     "llama3-vision-merged-vllm",
#     safe_serialization=True,          # <-- writes .safetensors
#     max_shard_size="4GB",
# )
# tokenizer.save_pretrained("llama3-vision-merged-vllm")  # <-- brings along the
#                                                         #     up-to-date template