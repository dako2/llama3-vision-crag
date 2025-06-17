from PIL import Image
import vllm
from typing import List

# Constants
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.95
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 1
TARGET_WIDTH, TARGET_HEIGHT = 960, 1280

def resize_images(images: List[Image.Image], target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT) -> List[Image.Image]:
    return [img.resize((target_width, target_height), Image.LANCZOS) if img.size != (target_width, target_height) else img for img in images]

# Initialize LLM
llm = vllm.LLM(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    max_model_len=MAX_MODEL_LEN,
    max_num_seqs=MAX_NUM_SEQS,
    trust_remote_code=True,
    dtype="auto", #fp16, bfloat16, half
    enforce_eager=True,
    limit_mm_per_prompt={"image": 2}
    
)

tokenizer = llm.get_tokenizer()

summarize_prompt = (
    "Identity the specific name of the object that the user is asking in the image. "
    "Don't answer the question itself but provide only the object identification that the user is asking {query}. "
    "If you are not sure, please respond 'I don't know' directly."
)

# Load and prepare image
pil_image = Image.open("../image.jpg")
if pil_image.mode != "RGB":
    pil_image = pil_image.convert("RGB")

queries = ['what is this brand?','what is this brand?']
images = resize_images([pil_image, pil_image])

# Format inputs
inputs, messages_batch = [], []
for query, image in zip(queries, images):
    messages = [{"role": "user", "content": 
                [
                    #{"type": "image"}, 
                    {"type": "text", "text": summarize_prompt.format(query=query)}
                ]}]
                
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs.append(
        {
            "prompt": formatted_prompt, 
            #"multi_modal_data": {"image": image}
        }
    )
    messages_batch.append(messages)


# <|begin_of_text|><|start_header_id|>user<|end_header_id|>

# <|image|>Describe this image in two sentences<|eot_id|><|start_header_id|>assistant<|end_header_id|>


# <|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

# Who are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>



print("\n============================= Formatted Prompt =============================")
print(formatted_prompt)
print("============================================================================\n")

# Generate
outputs = llm.generate(inputs, sampling_params=vllm.SamplingParams(
    temperature=0.01,
    top_p=0.85,
    max_tokens=30,
    skip_special_tokens=True,
    seed=42
))

# Display
summaries = [output.outputs[0].text.strip() for output in outputs]
for q, summary in zip(queries, summaries):
    print(f"Q: {q}\n→ Summary: {summary}\n")
