from typing import Dict, List, Any, Optional
import os
from PIL import Image

import torch
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline
from crag_web_result_fetcher import WebSearchResult
#from reranker import FastReRanker
import json
import re
import time 
# --- Florence object cropper (import from your utils) ---
#from florence_cropper import FlorenceObjectCropper
import unsloth
from unsloth import FastVisionModel

#from llm_logger import llm_logger
import spacy
from concurrent.futures import ThreadPoolExecutor

# fast_reranker = FastReRanker()
import vllm
# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 1
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
VLLM_TENSOR_PARALLEL_SIZE = 2
VLLM_GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 8
MAX_GENERATION_TOKENS = 75
NUM_SEARCH_RESULTS = 3
DEBUG_ENABLE = True
TARGET_WIDTH = 960
TARGET_HEIGHT = 1280
expected_size = (TARGET_WIDTH, TARGET_HEIGHT)

def format_input(args, tokenizer):
    messages, image = args
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return {"prompt": prompt, "multi_modal_data": {"image": image}}

# Load spaCy model once (outside function if possible)
nlp = spacy.load("en_core_web_sm")


# fast_reranker = FastReRanker()

# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 1
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
VLLM_TENSOR_PARALLEL_SIZE = 2
VLLM_GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 8
MAX_GENERATION_TOKENS = 75
NUM_SEARCH_RESULTS = 3
DEBUG_ENABLE = True


TARGET_WIDTH = 960
TARGET_HEIGHT = 1280
expected_size = (TARGET_WIDTH, TARGET_HEIGHT)
def resize_images(images: List[Image.Image], target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT) -> List[Image.Image]:
    resized_images = []
    for img in images:
        if img.size != (target_width, target_height):
            img = img.resize((target_width, target_height), Image.LANCZOS)
        resized_images.append(img)
    return resized_images

def safe_json_parse(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            raise ValueError("No valid JSON object found in response.")
    except json.JSONDecodeError as e:
        print("⚠️ JSON decode error:")
        print(text)
        raise e

class SimpleRAGAgent(BaseAgent):
    """
    SimpleRAGAgent demonstrates all the basic components you will need to create your 
    RAG submission for the CRAG-MM benchmark.
    """
    def __init__(
        self, 
        search_pipeline: UnifiedSearchPipeline, 
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", 
        max_gen_len: int = 64
    ):
        super().__init__(search_pipeline)
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        #self.initialize_models()
        self.initialize_models_full()

        #self.cropper = FlorenceObjectCropper(device="cpu", save_dir="temp")


    def initialize_models_full(self):
        print("🚀 Loading Unsloth LLaMA 3.2 Vision + LoRA fine-tuned adapter...")

        # Assuming your LoRA adapter is for the language part of the model
        # self.lora_request = LoRARequest(
        #     lora_name="llama3-vision-finetuned",
        #     lora_int_id=1,
        #     lora_local_path="./llama3-vision-finetuned/"
        # )

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
        self.tokenizer_vllm = self.llm.get_tokenizer()

        # # Step 2: Prepare for PEFT + load adapter
        # FastVisionModel.for_inference(self.model)  # allows LoRA merging at inference
        # self.model.load_adapter("llama3-vision-finetuned/")  # path to your saved LoRA adapter
        # self.model.eval()

        # # Move to device
        self.device = torch.device("cuda") # if torch.cuda.is_available() else "cpu"
        # self.model.to(self.device)

        print("✅ Model + LoRA adapter loaded and ready for inference.")


    def initialize_models(self):
        print("Loading base model and LoRA adapter from checkpoint...")

        # self.model, self.tokenizer = FastVisionModel.from_pretrained(
        #     "llama3-vision-finetuned",
        #     load_in_4bit=True,
        #     use_gradient_checkpointing="unsloth"
        # )

        # Load the base vision model
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name="unsloth/Llama-3.2-11B-Vision-Instruct",
            load_in_4bit=True,  # Optional: enables 4-bit quantization for efficiency
        )
        
        #self.model.load_adapter("llama3-vision-finetuned/", adapter_name="qa")
        #self.model.load_adapter("trainer_output/checkpoint-186", adapter_name="qa")
        self.model.eval()
        
        #FastVisionModel.for_inference(self.model)
        #self.model.load_adapter("trainer_output/checkpoint-186")

        print("✅ Loaded fine-tuned model with LoRA for inference.")


    def get_batch_size(self) -> int:
        return AICROWD_SUBMISSION_BATCH_SIZE

    def generate_image_captions(
        self,
        images: List[Image.Image],
        queries: List[str]
    ) -> List[str]:
 
        messages_batch = []
        for image, query in zip(images, queries):
            SYSTEM_PROMPT = "You provide specific object identification in an image given a user's query. Don't answer the question itself but only provide the object name. Be concise in one sentence."
            USER_QUERY = f"Analyze the image and this question -- {query}. Identify the object name in the image for web search."

            #SYSTEM_PROMPT = """You help analyze the image to find out what's the object the user is asking in the query and rephrase the question with better context for web search."""
            #USER_QUERY= """find me the specific object identity that I'm asking in the image: {query}."""
            #USER_QUERY = f"\nUser query: {query}"
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_QUERY}]},
            ]
            messages_batch.append(messages)

        captions = self.inference_unsloth(images, messages_batch, max_tokens=256, load_adapter=False)

        # captions = ""
        # # Extract and print multiple completions
        # for i, output in enumerate(outputs):
        #     print(f"\n--- Candidate {i+1} ---\n")
        #     captions += output.outputs[0].text.strip()
            
        return captions

    def crop_images(
        self,
        images: List[Image.Image],
        captions: List[str]
    ) -> List[Optional[Image.Image]]:
        cropped_images = []
        for image, caption in zip(images, captions):
            cropped_paths = self.cropper.detect_and_crop(image=image, target_label=caption)
            cropped = Image.open(cropped_paths[0]) if cropped_paths else None
            cropped_images.append(cropped)
        return cropped_images

    def search_candidates(
        self,
        images: List[Image.Image],
        captions: List[str],
        queries: List[str]
    ) -> List[List[str]]:
        all_candidates = []
        for image, caption, query in zip(images, captions, queries):
            candidate_contexts = []
            if image:
                image_search_results = self.search_pipeline(image, k=2)
                for item in image_search_results:
                    candidate_contexts.append(str(item))
            web_search_results = self.search_pipeline(query + caption, k=3)

            for hit in web_search_results:
                hit = WebSearchResult(hit)
                candidate_contexts.append(hit["page_content"][:500])
            all_candidates.append(candidate_contexts)
        return all_candidates

    def batch_summarize_and_crop_images(
        self,
        images: List[Image.Image],
        queries: List[str]
    ) -> List[str]:

        print("\n\n##Queries:", queries)
        # Step 1: Caption images
        captions = self.generate_image_captions(images, queries)
        print("##Caption:", captions)

        # Step 2: Crop with FlorenceObjectCropper
        retrieved_context = []
        for orig_img, caption, query in zip(images, captions, queries):
            
            candidate_sentences = []
            search_str = query + caption
            
            # cropped_paths = self.cropper.detect_and_crop(orig_img, query)
            # if cropped_paths:
            #     print("image cropping done")
            #     print("searching the image...")
            #     cropped = Image.open(cropped_paths[0])
            #     image_search_results = self.search_pipeline(cropped, k=2)
            #     for item in image_search_results:
            #         candidate_sentences.append(str(item))
            # else:
            #     print("no cropped image")

            # Web search
            
            # Web search (can uncomment image crop if needed later)

            results = self.search_pipeline(search_str, k=5)

            for i, result in enumerate(results):
                #result = WebSearchResult(result)
                snippet = result.get('page_snippet', '')
                if not snippet:
                    continue
                # # Sentence splitting with spaCy
                # doc = nlp(snippet)
                # for sent in doc.sents:
                #     sentence = sent.text.strip()
                #     if sentence:
                #         candidate_sentences.append(f"[Info {i+1}] {sentence}")

                # raw_sentences = re.split(r'(?<=[.!?])\s+', snippet.strip())
                # for sentence in raw_sentences:
                #     if sentence:
                #         candidate_sentences.append(f"[Info {i+1}] {sentence.strip()}")
                candidate_sentences.append(snippet)

            print(candidate_sentences)

            # for hit in results:
            #     hit = WebSearchResult(hit)
            #     candidate_contexts.append(hit["page_content"][:500])
                
            #print("web search results", candidate_contexts)

            # Rerank top 3 sentences most relevant to the query
            # ranked = fast_reranker.rerank(search_str, candidate_sentences, top_k=5)
            # print("ranked",ranked)

            context_str = ""
            # for context, score in ranked:
            #     context_str += f"Score: {score:.4f} - Context: {context}\n"
            for context in candidate_sentences:
                context_str += f"Context: {context}\n"

            context_str += f"\n[Image caption]: {search_str}"
            retrieved_context.append(context_str)


            #print("reranking results: ", context_str)
        return retrieved_context

    def inference(
        self, 
        images,
        messages_batch: List[str],
        max_tokens: int = MAX_GENERATION_TOKENS
    ) -> List[str]:
        responses = []

        for messages, image in zip(messages_batch, images):
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            inputs["multi_modal_data"] = {"image": image}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.9,
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(decoded)

        return responses

    def inference_unsloth(
        self, 
        images,
        messages_batch: List[str],
        max_tokens: int = MAX_GENERATION_TOKENS,
        load_adapter: bool = False
    ) -> List[str]:

        #load_adapter = False

        if not load_adapter:         
            self.model.disable_adapters()       
            self.model.eval()
            print("disable adapter")
            
        outputs = []
        for image, messages in zip(images, messages_batch):
            new_messages = convert_messages_from_vllm_to_unsloth_format(messages)
            # new_messages = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {"type": "image"},
            #             {"type": "text", "text": SYSTEM_PROMPT+guidance_text},
            #         ],
            #     }
            # ]
            prompt = self.tokenizer.apply_chat_template(new_messages, add_generation_prompt=True)
            inputs = self.tokenizer(
                image,
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_new_tokens=max_tokens,
                #max_length=8192,
            ).to(self.device)

            output = self.model.generate(
                **inputs,
                temperature=0.1,
                max_new_tokens=max_tokens,
                use_cache=False,
            )

            answer = (
                self.tokenizer.decode(output[0])
                .split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                .split("<|eot_id|>")[0]
                .strip()
            )
            outputs.append(answer)
        
        if not load_adapter:
            self.model.set_adapter("qa")
            self.model.eval()
            print("loading adapter")

        return outputs 
        
    def batch_generate_response(
            self,
            queries: List[str],
            images: List[Image.Image],
            message_histories: List[List[Dict[str, Any]]],
        ) -> List[str]: # This return type annotation will need to change

        images = resize_images(images)

        t1 = time.time()
        search_results_batch = self.batch_summarize_and_crop_images(
            images, queries
        )
        print("Processed batch of search_results_batch queries with RAG: %.2f seconds"%(time.time()-t1))

        messages_batch = []

        for idx, (query, image, message_history, rag_context) in enumerate(
            zip(queries, images, message_histories, search_results_batch)
        ):
            system_prompt = """You are a helpful and honest assistant. Please, respond concisely and truthfully in {token_limit} words or less."""
            #user_prompt = """Context information is below. {context_str} Given the context information and using your prior knowledge, please provide your answer in concise style. End your answer with a period. Answer the question in one line only. Question: {query_str} Answer: """
            user_prompt = (
                "Given the context below and the image, answer the question truthfully in one line. "
                "Use context to support your answer explicitly. If insufficient information is available, say so.\n\n"
                "Context: {context_str}\n\n"
                "Question: {query_str}\nAnswer:"
            )

            messages = [
                {"role": "system", "content": system_prompt.format(token_limit=65)},
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "user","content": user_prompt.format(context_str = rag_context, query_str = query)}
            ]
            
            if message_history:
                messages = messages + message_history
                
            # reflections = [
            #     "Reflect on your answer. Was it grounded in the context?",
            #     "Was the answer factually supported by the provided context?",
            #     "How confident are you in the correctness of the answer?"
            # ]

            # batched_inputs = [
            #     {"prompt": base_prompt + "\n" + reflection, "multi_modal_data": {"image": image}}
            #     for reflection in reflections
            # ]
        
            messages_batch.append(messages)
        
        t0 = time.time()
        responses = self.inference_unsloth(images,
            messages_batch, max_tokens=75, load_adapter=True
        )
        print("total inference time for final answer generation batch: %.2f seconds"%(time.time()-t0))
        

        system_prompt_2 = (
            "Reflect on your previous answer. Was it grounded in the provided context and correct? "
            "If not, revise it to ensure factual accuracy. Do not hallucinate. End with a period."
        )

        print("responses", responses)

        # postprocess_messages(responses, messages_batch, queries)
        # --- MODIFICATION: RETURN THE CONTEXT AS WELL ---
        return responses # Assuming search_results_batch is List[str] where each string is the combined context for a query

def convert_messages_from_vllm_to_unsloth_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts a VLLM-style chat message history into a Unsloth-compatible format.
    - Merges system/user/assistant messages into a single `user` prompt.
    - Appends image and formatted text.
    """
    image = None
    parts = []

    for m in messages:
        role = m["role"]
        content = m["content"]

        if role == "user" and isinstance(content, list):
            # Look for image in multimodal messages
            for part in content:
                if part.get("type") == "image":
                    image = {"type": "image"}
                elif part.get("type") == "text":
                    parts.append(part["text"])
        elif role in {"system", "user", "assistant"} and isinstance(content, str):
            # Just append the text parts from all roles except image parts
            parts.append(f"[{role.upper()}] {content}")

    merged_text = "\n".join(parts).strip()

    unsloth_message = {
        "role": "user",
        "content": []
    }

    if image:
        unsloth_message["content"].append(image)

    unsloth_message["content"].append({"type": "text", "text": merged_text})

    return [unsloth_message]


def postprocess_messages(responses, messages_batch, queries):

    for response, messages_data, query in zip(responses, messages_batch, queries):

        system_prompt = messages_data[0]["content"]
        user_text = messages_data[2]["content"]

        new_format = {
            "query": query,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ""},
                        {"type": "text", "text": system_prompt + "\n" + user_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": response}
                    ]
                }
            ]
        }

        #llm_logger.log(new_format)

        