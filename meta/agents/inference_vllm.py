from typing import Dict, List, Any
import os

import vllm
from PIL import Image

#### Please ensure that when you submit, VLLM_TENSOR_PARALLEL_SIZE=1. 


VLLM_TENSOR_PARALLEL_SIZE = 2
VLLM_GPU_MEMORY_UTILIZATION = 0.85 


# These are model specific parameters to get the model to run on a single NVIDIA L40s GPU
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 1
MAX_GENERATION_TOKENS = 75

# Number of search results to retrieve
NUM_SEARCH_RESULTS = 5

# GPU utilization settings 
# Change VLLM_TENSOR_PARALLEL_SIZE during local runs based on your available GPUs
# For example, if you have 2 GPUs on the server, set VLLM_TENSOR_PARALLEL_SIZE=2. 
# You may need to uncomment the following line to perform local evaluation with VLLM_TENSOR_PARALLEL_SIZE>1. 
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

class VLLM_IMAGE():
    def __init__(self, model_name):
        self.model_name = model_name
        # Initialize the model with vLLM
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={
                "image": 1 
            } # In the CRAG-MM dataset, every conversation has at most 1 image
        )
        self.tokenizer = self.llm.get_tokenizer()

    def inference(self, messages_batch, images, MAX_GENERATION_TOKENS = 70):
        # messages = [
        #     {"role": "system", "content": "You are a helpful assistant that accurately describes images. Your responses are subsequently used to perform a web search to retrieve the relevant information about the image."},
        #     {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": summarize_prompt}]},
        # ]
        inputs = []
        for messages, image in zip(messages_batch, images):
            # Format prompt using the tokenizer
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            print(formatted_prompt)
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
        
        # Generate summaries in a single batch call
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=MAX_GENERATION_TOKENS,  # Short summary only
                skip_special_tokens=True
            )
        )
        
        # Extract and clean summaries
        results = [output.outputs[0].text.strip() for output in outputs]
        return results