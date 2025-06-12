from typing import Dict, List, Any
import os
import unsloth
from unsloth import FastVisionModel
import torch
from PIL import Image
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

from crag_web_result_fetcher import WebSearchResult

import json
from pathlib import Path
import time 

max_tokens = 75

# oyiyi muted
# def convert_messages_from_vllm_to_unsloth_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     Converts a VLLM-style chat message history into a Unsloth-compatible format.
#     - Merges system/user/assistant messages into a single `user` prompt.
#     - Appends image and formatted text.
#     """
#     image = None
#     parts = []

#     for m in messages:
#         role = m["role"]
#         content = m["content"]

#         if role == "user" and isinstance(content, list):
#             # Look for image in multimodal messages
#             for part in content:
#                 if part.get("type") == "image":
#                     image = {"type": "image"}
#                 elif part.get("type") == "text":
#                     parts.append(part["text"])
#         elif role in {"system", "user", "assistant"} and isinstance(content, str):
#             # Just append the text parts from all roles except image parts
#             parts.append(f"[{role.upper()}] {content}")

#     merged_text = "\n".join(parts).strip()

#     unsloth_message = {
#         "role": "user",
#         "content": []
#     }

#     if image:
#         unsloth_message["content"].append(image)

#     unsloth_message["content"].append({"type": "text", "text": merged_text})

#     return [unsloth_message]

def convert_messages_from_vllm_to_unsloth_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts VLLM-style message history into Unsloth-compatible format.
    Ensures image comes first in multimodal content.
    """
    unsloth_message = {
        "role": "user",
        "content": []
    }

    found_image = False
    text_parts = []

    for m in messages:
        role = m["role"]
        content = m["content"]

        if role == "user" and isinstance(content, list):
            for part in content:
                if part.get("type") == "image":
                    if not found_image:
                        unsloth_message["content"].append({"type": "image"})
                        found_image = True
                elif part.get("type") == "text":
                    text_parts.append(part["text"])
        elif isinstance(content, str):
            text_parts.append(f"[{role.upper()}] {content}")

    merged_text = "\n".join(text_parts).strip()

    # Add text after image, always
    if merged_text:
        unsloth_message["content"].append({"type": "text", "text": merged_text})

    return [unsloth_message]


# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 2

# GPU utilization settings 
# Change VLLM_TENSOR_PARALLEL_SIZE during local runs based on your available GPUs
# For example, if you have 2 GPUs on the server, set VLLM_TENSOR_PARALLEL_SIZE=2. 
# You may need to uncomment the following line to perform local evaluation with VLLM_TENSOR_PARALLEL_SIZE>1. 
#os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

#### Please ensure that when you submit, VLLM_TENSOR_PARALLEL_SIZE=1. 
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.95

# These are model specific parameters to get the model to run on a single NVIDIA L40s GPU
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2
MAX_GENERATION_TOKENS = 75

# Number of search results to retrieve
NUM_SEARCH_RESULTS = 3

class SimpleRAGAgent(BaseAgent):
    """
    SimpleRAGAgent demonstrates all the basic components you will need to create your 
    RAG submission for the CRAG-MM benchmark.
    Note: This implementation is not tuned for performance, and is intended for demonstration purposes only.
    
    This agent enhances responses by retrieving relevant information through a search pipeline
    and incorporating that context when generating answers. It follows a two-step approach:
    1. First, batch-summarize all images to generate effective search terms
    2. Then, retrieve relevant information and incorporate it into the final prompts
    
    The agent leverages batched processing at every stage to maximize efficiency.
    
    Note:
        This agent requires a search_pipeline for RAG functionality. Without it,
        the agent will raise a ValueError during initialization.
    
    Attributes:
        search_pipeline (UnifiedSearchPipeline): Pipeline for searching relevant information.
        model_name (str): Name of the Hugging Face model to use.
        max_gen_len (int): Maximum generation length for responses.
        llm (vllm.LLM): The vLLM model instance for inference.
        tokenizer: The tokenizer associated with the model.
    """

    def __init__(
        self, 
        search_pipeline: UnifiedSearchPipeline, 
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", 
        max_gen_len: int = 64
    ):
        """
        Initialize the RAG agent with the necessary components.
        
        Args:
            search_pipeline (UnifiedSearchPipeline): A pipeline for searching web and image content.
                Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
                      Hence, this implementation of the RAG agent is not suitable for Task 1 (Single-source Augmentation).
            model_name (str): Hugging Face model name to use for vision-language processing.
            max_gen_len (int): Maximum generation length for model outputs.
            
        Raises:
            ValueError: If search_pipeline is None, as it's required for RAG functionality.
        """
        super().__init__(search_pipeline)
        
        if search_pipeline is None:
            raise ValueError("Search pipeline is required for RAG agent")
            
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        
        # ----------------------------- NEW -----------------------------
        # Flat list of every message/event (keeps order)
        self.messages: List[Dict[str, Any]] = []
        # Starting index of each chat session inside `self.messages`
        self.session_indices: List[int] = [0]
        self._current_session_id: int = 0
        # ----------------------------------------------------------------
 
        self.initialize_models()
        
    
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # self.model.to(self.device)
        # Only call .to(self.device) if model is NOT loaded in 4bit or 8bit mode
        if not getattr(self.model, "is_loaded_in_4bit", False) and not getattr(self.model, "is_loaded_in_8bit", False):
            self.model.to(self.device)
        
        #self.model.load_adapter("llama3-vision-finetuned/", adapter_name="qa")
        #self.model.load_adapter("trainer_output/checkpoint-186", adapter_name="qa")
        self.model.eval()
        
        #FastVisionModel.for_inference(self.model)
        #self.model.load_adapter("trainer_output/checkpoint-186")

        print("✅ Loaded fine-tuned model with LoRA for inference.")

    def _log_for_sft(
        self,
        session_id: str,
        assistant_answer: str,
        history=None,
        file_path: str="sft_caption_data.jsonl",
    ):
        """
        Append one training example in the minimal format expected by
        Llama-3 Vision fine-tuning (session_id + messages).

        Parameters
        ----------
        session_id : str
        query      : str               # the user question
        assistant_answer : str         # model response
        history    : list[dict] | None # previous turns (same format you already use)
        file_path  : Str              # where to append (default: ./sft_data.jsonl)
        """
        row = {
            "session_id": session_id,
            "messages": (history or []) + [
                {"role": "assistant", "content": assistant_answer},
            ],
        }
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling batch_generate_response.
        
        The evaluator uses this value to determine how many queries to send in each batch.
        Valid values are integers between 1 and 16.
        
        Returns:
            int: The batch size, indicating how many queries should be processed together 
                 in a ssingle batch.
        """
        return AICROWD_SUBMISSION_BATCH_SIZE
    
    def batch_summarize_images(self, session_ids, queries, images: List[Image.Image]) -> List[str]:
        """
        Generate brief summaries for a batch of images to use as search keywords.
        
        This method efficiently processes all images in a single batch call to the model,
        resulting in better performance compared to sequential processing.
        
        Args:
            images (List[Image.Image]): List of images to summarize.
            
        Returns:
            List[str]: List of brief text summaries, one per image.
        """
        # Prepare image summarization prompts in batch
        summarize_prompt = "You provide specific object identification in an image given a user's query. Don't answer the question itself but only provide the object name. Be concise in one sentence. If you are not sure, just reply 'i don't know'."
        
        inputs = []
        outputs = []
        messages_batch = []
        for query, image in zip(queries, images):
            messages = [
                # {"role": "system", "content": summarize_prompt},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{summarize_prompt} Analyze the image and this question -- {query}. Identify the object name in the image for web search."}]},
            ]
            
            # Format prompt using the tokenizer
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
            messages_batch.append(messages)
        
            new_messages = convert_messages_from_vllm_to_unsloth_format(messages)
            prompt = self.tokenizer.apply_chat_template(new_messages, add_generation_prompt=True)
            input_tensor = self.tokenizer(
                image,
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_new_tokens=max_tokens,
                max_length=8192,
            ).to(self.device)

            output = self.model.generate(
                **input_tensor,
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
        
        # Extract and clean summaries
        summaries = outputs#[output.outputs[0].text.strip() for output in outputs]
        print(f"Generated {len(summaries)} image summaries: {summaries}")

        for session_id, answer, history in zip(session_ids, summaries, messages_batch): 
            self._log_for_sft(session_id, answer, history, file_path="sft_caption_data.jsonl")

        return summaries


    def zero_shots(self, session_ids, queries, images: List[Image.Image]) -> List[str]:
        """
        Generate brief summaries for a batch of images to use as search keywords.
        
        This method efficiently processes all images in a single batch call to the model,
        resulting in better performance compared to sequential processing.
        
        Args:
            images (List[Image.Image]): List of images to summarize.
            
        Returns:
            List[str]: List of brief text summaries, one per image.
        """
        # Prepare image summarization prompts in batch
        # summarize_prompt = "Honestly answer the user's tricky question based on the image. Be concise in one sentence. If you are not sure, just reply 'i don't know'"
        summarize_prompt = "Honestly answer the user's tricky question based on the image. Be concise in one sentence. If you are not sure, just reply 'i don't know'."
        # 2024: You are a helpful and honest assistant. Please, respond concisely and truthfully in {token_limit} words or less. If you are not sure about the query, answer I don’t know. There is no need to explain the reasoning behind your answers. "
        
        inputs = []
        outputs = []
        messages_batch = []
        for query, image in zip(queries, images):
            messages = [
                # {"role": "system", "content": summarize_prompt}, # Oyiyi
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{summarize_prompt}{query}."}]},
            ]
            
            # Format prompt using the tokenizer
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
            messages_batch.append(messages)
        
        
            new_messages = convert_messages_from_vllm_to_unsloth_format(messages)
            prompt = self.tokenizer.apply_chat_template(new_messages, add_generation_prompt=True)
            input_tensor = self.tokenizer(
                image,
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_new_tokens=max_tokens,
                max_length=8192,
            ).to(self.device)

            output = self.model.generate(
                **input_tensor,
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
        
        # Extract and clean summaries
        summaries = outputs#[output.outputs[0].text.strip() for output in outputs]
        print(f"Generated {len(summaries)} image summaries: {summaries}")

        for session_id, answer, history in zip(session_ids, summaries, messages_batch): 
            self._log_for_sft(session_id, answer, history, file_path="sft_caption_data_case_0_zero_shot.jsonl")

        return summaries
    
    def batch_images_search(self, session_ids, images: List[Image.Image]) -> List[str]:
        """
        Generate brief summaries for a batch of images to use as search keywords.
        
        This method efficiently processes all images in a single batch call to the model,
        resulting in better performance compared to sequential processing.
        
        Args:
            images (List[Image.Image]): List of images to summarize.
            
        Returns:
            List[str]: List of brief text summaries, one per image.
        """
        
        # Retrieve relevant information for each query
        search_results_batch = []
        for image in images:
            results = self.search_pipeline(image, k=1)
            search_results_batch.append(results)

        return search_results_batch
    

    def prepare_rag_enhanced_inputs(
        self, 
        session_ids,
        queries: List[str], 
        images: List[Image.Image], 
        image_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]]
    ) -> List[dict]:
        """
        Prepare RAG-enhanced inputs for the model by retrieving relevant information in batch.
        
        This method:
        1. Uses image summaries combined with queries to perform effective searches
        2. Retrieves contextual information from the search_pipeline
        3. Formats prompts incorporating this retrieved information
        
        Args:
            queries (List[str]): List of user questions.
            images (List[Image.Image]): List of images to analyze.
            image_summaries (List[str]): List of image summaries for search.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories.
            
        Returns:
            List[dict]: List of input dictionaries ready for the model.
        """
        # Batch process search queries
        search_results_batch = []
        
        # Create combined search queries for each image+query pair
        search_queries = [f"{query} {summary}" for query, summary in zip(queries, image_summaries)]
        #TODO
        
        # Retrieve relevant information for each query
        for i, search_query in enumerate(search_queries):
            results = self.search_pipeline(search_query, k=NUM_SEARCH_RESULTS)
            search_results_batch.append(results)

        return search_results_batch

    def prepare_rag_enhanced_inputs_with_rephrase(
        self, 
        session_ids,
        queries: List[str], 
        images: List[Image.Image], 
        image_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]]
    ) -> List[dict]:
        """
        Prepare RAG-enhanced inputs for the model by retrieving relevant information in batch.
        
        This method:
        1. Uses image summaries combined with queries to perform effective searches
        2. Retrieves contextual information from the search_pipeline
        3. Formats prompts incorporating this retrieved information
        
        Args:
            queries (List[str]): List of user questions.
            images (List[Image.Image]): List of images to analyze.
            image_summaries (List[str]): List of image summaries for search.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories.
            
        Returns:
            List[dict]: List of input dictionaries ready for the model.
        """

        # Prepare image summarization prompts in batch
        summarize_prompt = "Be helpful assistant on web search."
        
        inputs = []
        outputs = []
        messages_batch = []
        for query, caption, image in zip(queries, image_summaries, images):
            messages = [
                # {"role": "system", "content": summarize_prompt}, # Oyiyi
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{summarize_prompt} Given the image caption and user's query, clarify the question into one sentence for better keywords for web search. Question:{query}, Image Caption:{caption}"}]},
            ]
            
            # Format prompt using the tokenizer
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
            messages_batch.append(messages)
        
            new_messages = convert_messages_from_vllm_to_unsloth_format(messages)
            prompt = self.tokenizer.apply_chat_template(new_messages, add_generation_prompt=True)
            input_tensor = self.tokenizer(
                image,
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_new_tokens=max_tokens,
                max_length=8192,
            ).to(self.device)

            output = self.model.generate(
                **input_tensor,
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
        
        # Extract and clean summaries
        summaries =outputs #[output.outputs[0].text.strip() for output in outputs]
        print(f"Generated {len(summaries)} image summaries: {summaries}")

        # Batch process search queries
        search_results_batch = []
        
        # Create combined search queries for each image+query pair
        search_queries = [f"{query} {summary}" for query, summary in zip(queries, image_summaries)]
        
        # Retrieve relevant information for each query
        for i, search_query in enumerate(search_queries):
            results = self.search_pipeline(search_query, k=NUM_SEARCH_RESULTS)
            search_results_batch.append(results)

        return search_results_batch


        
    def inference(
        self, 
        session_ids,
        search_results_batch,
        queries: List[str], 
        images: List[Image.Image], 
        image_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]],
        save_sft_data_path: str,
    ) -> List[dict]:

        # Prepare formatted inputs with RAG context for each query
        inputs = []
        messages_batch = []


        outputs = [] 

        
        for idx, (query, image, message_history, search_results, caption) in enumerate(
            zip(queries, images, message_histories, search_results_batch, image_summaries)
        ):
            # Create system prompt with RAG guidelines
            # SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions about the provided image."
            #                "Keep your response concise and to the point. If you don't know the answer, respond with 'I don't know'.")
            
            system_prompt = """You are in factual Q&A competition. Please respond concisely and truthfully in 65 words or less. If you don't know the answer, respond with 'I don't know'."""
            #user_prompt = """Context information is below. {context_str} Given the context information and using your prior knowledge, please provide your answer in concise style. End your answer with a period. Answer the question in one line only. Question: {query_str} Answer: """
            user_prompt = (
                "Given the context below and the image, answer the question truthfully in one line. "
                "Use context to support your answer explicitly. If insufficient information is available, say so.\n\n"
                "##Image Caption: {caption}\n"
                "##Some Context: {context_str}\n"
                "##Question: {query_str}\n"
                "##Answer:"
            )

            # Add retrieved context if available
            rag_context = ""
            if search_results:
                for i, result in enumerate(search_results):
                    # WebSearchResult is a helper class to get the full page content of a web search result.
                    #
                    # It first checks if the page content is already available in the cache. If not, it fetches  
                    # the full page content and caches it.
                    #
                    # WebSearchResult adds `page_content` attribute to the result dictionary where the page 
                    # content is stored. You can use it like a regular dictionary to fetch other attributes.
                    #
                    # result["page_content"] for complete page content, this is available only via WebSearchResult
                    # result["page_url"] for page URL
                    # result["page_name"] for page title
                    # result["page_snippet"] for page snippet
                    # result["score"] relavancy with the search query
                    result = WebSearchResult(result)
                    snippet = result.get('page_snippet', '')
                    if snippet:
                        rag_context += f"[Info {i+1}] {snippet}\n\n"
                
            # Structure messages with image and RAG context
            messages = [
                # {"role": "system", "content": system_prompt}, # Oyiyi
                #{"role": "user", "content": [{"type": "image"}]}
            ]
            
            # Add conversation history for multi-turn conversations
            if message_history:
                messages = messages + message_history
            
            # Add the current query # Oyiyi
            messages.append({"role": "user", "content": f"{system_prompt}\n" + user_prompt.format(caption=caption, context_str=rag_context, query_str=query)})
            

            new_messages = convert_messages_from_vllm_to_unsloth_format(messages)
            
            # Oyiyi replaced below 2
            # prompt = self.tokenizer.apply_chat_template(new_messages, add_generation_prompt=True)
            # input_tensor = self.tokenizer(
            #     image,
            #     prompt,
            #     add_special_tokens=False,
            #     return_tensors="pt",
            #     truncation=True,
            #     max_new_tokens=max_tokens,
            #     #max_length=8192,
            # ).to(self.device)

            # Format with chat template
            prompt = self.tokenizer.apply_chat_template(
                new_messages,
                add_generation_prompt=True,
                tokenize=False
            )

            # Ensure image is properly included and tokenizer receives both inputs
            input_tensor = self.tokenizer(
                text=prompt,
                image=image,
                return_tensors="pt",
                truncation=True,
                max_new_tokens=max_tokens,
                max_length=8192,
            ).to(self.device)


            output = self.model.generate(
                **input_tensor,
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

        # Extract and return the generated responses
        responses = outputs#[output.outputs[0].text for output in outputs]
        print(f"Successfully generated {len(responses)} responses")
        
        for session_id, answer, history in zip(session_ids, responses, messages_batch): 
            self._log_for_sft(session_id, answer, history, file_path=save_sft_data_path) #"sft_response_data_case_2.jsonl"

        return responses

    def inference_w_image(
        self, 
        session_ids,
        search_results_batch,
        queries: List[str], 
        images: List[Image.Image], 
        image_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]],
        save_sft_data_path: str,
    ) -> List[dict]:

        # Prepare formatted inputs with RAG context for each query
        inputs = []
        outputs = []
        messages_batch = []
        for idx, (query, image, message_history, search_results, caption) in enumerate(
            zip(queries, images, message_histories, search_results_batch, image_summaries)
        ):
            # Create system prompt with RAG guidelines
            # SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions about the provided image."
            #                "Keep your response concise and to the point. If you don't know the answer, respond with 'I don't know'.")
            
            system_prompt = """You are in factual Q&A competition. Please respond concisely and truthfully in 65 words or less. If you don't know the answer, respond with 'I don't know'."""
            #user_prompt = """Context information is below. {context_str} Given the context information and using your prior knowledge, please provide your answer in concise style. End your answer with a period. Answer the question in one line only. Question: {query_str} Answer: """
            user_prompt = (
                "Given the context below and the image, answer the question truthfully in one line. "
                "Use context to support your answer explicitly. If insufficient information is available, say so.\n\n"
                "##Image Caption: {caption}\n"
                "##Some Context: {context_str}\n"
                "##Question: {query_str}\n"
                "##Answer:"
            )

            # Add retrieved context if available
            rag_context = ""
            if search_results:
                for i, result in enumerate(search_results):
                    # WebSearchResult is a helper class to get the full page content of a web search result.
                    #
                    # It first checks if the page content is already available in the cache. If not, it fetches  
                    # the full page content and caches it.
                    #
                    # WebSearchResult adds `page_content` attribute to the result dictionary where the page 
                    # content is stored. You can use it like a regular dictionary to fetch other attributes.
                    #
                    # result["page_content"] for complete page content, this is available only via WebSearchResult
                    # result["page_url"] for page URL
                    # result["page_name"] for page title
                    # result["page_snippet"] for page snippet
                    # result["score"] relavancy with the search query
                    result = WebSearchResult(result)
                    snippet = result.get('page_snippet', '')
                    if snippet:
                        rag_context += f"[Info {i+1}] {snippet}\n\n"
                
            # Structure messages with image and RAG context
            messages = [
                # {"role": "system", "content": system_prompt}, # Oyiyi
                {"role": "user", "content": [{"type": "image"}]}
            ]
            
            # Add conversation history for multi-turn conversations
            if message_history:
                messages = messages + message_history
            
            # Add the current query
            messages.append({"role": "user", "content": f"{system_prompt}\n" + user_prompt.format(caption=caption, context_str=rag_context, query_str=query)})
            
            # Apply chat template
            
            new_messages = convert_messages_from_vllm_to_unsloth_format(messages)
            prompt = self.tokenizer.apply_chat_template(new_messages, add_generation_prompt=True)
            input_tensor = self.tokenizer(
                image,
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_new_tokens=max_tokens,
                max_length=8192,
            ).to(self.device)

            output = self.model.generate(
                **input_tensor,
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

        # Extract and return the generated responses
        responses = outputs #[output.outputs[0].text for output in outputs]
        print(f"Successfully generated {len(responses)} responses")
        
        for session_id, answer, history in zip(session_ids, responses, messages_batch): 
            self._log_for_sft(session_id, answer, history, file_path=save_sft_data_path) #"sft_response_data_case_2.jsonl"

        return responses

    def batch_generate_response(
        self,
        session_ids, 
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """
        Generate RAG-enhanced responses for a batch of queries with associated images.
        
        This method implements a complete RAG pipeline with efficient batch processing:
        1. First batch-summarize all images to generate search terms
        2. Then retrieve relevant information using these terms
        3. Finally, generate responses incorporating the retrieved context
        
        Args:
            queries (List[str]): List of user questions or prompts.
            images (List[Image.Image]): List of PIL Image objects, one per query.
                The evaluator will ensure that the dataset rows which have just
                image_url are populated with the associated image.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories,
                one per query. Each history is a list of message dictionaries with
                'role' and 'content' keys in the following format:
                
                - For single-turn conversations: Empty list []
                - For multi-turn conversations: List of previous message turns in the format:
                  [
                    {"role": "user", "content": "first user message"},
                    {"role": "assistant", "content": "first assistant response"},
                    {"role": "user", "content": "follow-up question"},
                    {"role": "assistant", "content": "follow-up response"},
                    ...
                  ]
                
        Returns:
            List[str]: List of generated responses, one per input query.
        """
        print(f"Processing batch of {len(queries)} queries with RAG")
        t0 = time.time()

        
        responses = self.zero_shots(session_ids, queries, images) # case 0: no caption, just image, no context
        print("Case0",responses)

        # # Step 1: Batch summarize all images for search terms
        image_summaries = self.batch_summarize_images(session_ids, queries, images)
        
        # # Step 2: Prepare RAG-enhanced inputs in batch
        # search_results = self.prepare_rag_enhanced_inputs(session_ids, 
        #     queries, images, image_summaries, message_histories
        # )
        
        # responses = self.inference(session_ids, search_results, # case 2: context from web search, image captions, no image
        #     queries, images, image_summaries, message_histories, save_sft_data_path="sft_response_data_case_2_web_search_only.jsonl"
        # )
        # print("Case2",responses)

        # image_search_results = self.batch_images_search(session_ids, images)

        # responses = self.inference(session_ids, image_search_results, # case 3: context from image search, image captions, no image
        #     queries, images, image_summaries, message_histories, save_sft_data_path="sft_response_data_case_3_image_search_only.jsonl"
        # )
        # print("Case3",responses)

        # responses = self.inference(session_ids, len(session_ids)*[''], # case 1: no context from search, just image captions, no image
        #     queries, images, image_summaries, message_histories, save_sft_data_path="sft_response_data_case_1.jsonl"
        # )
        # print("Case1",responses)

        search_results = self.prepare_rag_enhanced_inputs_with_rephrase(session_ids, 
            queries, images, image_summaries, message_histories
        )
        
        responses = self.inference(session_ids, search_results, # case 5: context from web search (with rephrased keywords), image captions, on image
            queries, images, image_summaries, message_histories, save_sft_data_path="validation_sft_response_data_case_5_web_search_rephrase.jsonl"
        )
        print("Case5",responses)

        responses = self.inference_w_image(session_ids, search_results, # case 5: context from web search (with rephrased keywords), image captions, on image
            queries, images, image_summaries, message_histories, save_sft_data_path="validation_sft_response_data_case_5b_web_search_rephrase_w_image.jsonl"
        )
        print("Case5b",responses)

        # combined_results = []
        # for image_result, web_research in zip(image_search_results, search_results):
        #     combined_results.append(image_result+web_research)

        # responses = self.inference(session_ids, combined_results, # case 4: context from image search + web search, image captions, on image
        #     queries, images, image_summaries, message_histories, save_sft_data_path="sft_response_data_case_4_image_web.jsonl"
        # )
        # print("Case4",responses)
        # print(f"Processing batch of queries with {time.time()-t0} seconds")
        return responses
