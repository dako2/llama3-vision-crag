from typing import Dict, List, Any
import os

import torch
from PIL import Image
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

from crag_web_result_fetcher import WebSearchResult
import vllm

import json
from pathlib import Path


# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 8

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
MAX_NUM_SEQS = 8
MAX_GENERATION_TOKENS = 75

# Number of search results to retrieve
NUM_SEARCH_RESULTS = 5

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
        """
        Initialize the vLLM model and tokenizer with appropriate settings.
        
        This configures the model for vision-language tasks with optimized
        GPU memory usage and restricts to one image per prompt, as 
        Llama-3.2-Vision models do not handle multiple images well in a single prompt.
        
        Note:
            The limit_mm_per_prompt setting is critical as the current Llama vision models
            struggle with multiple images in a single conversation.
            Ref: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/discussions/43#66f98f742094ed9e5f5107d4
        """
        print(f"Initializing {self.model_name} with vLLM...")
        
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

        print("Models loaded successfully")
    
    def _log_for_sft(
        self,
        session_id: str,
        assistant_answer: str,
        history=None,
        file_path: Path = Path("sft_data.jsonl"),
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
        file_path  : Path              # where to append (default: ./sft_data.jsonl)
        """
        row = {
            "session_id": session_id,
            "messages": (history or []) + [
                {"role": "assistant", "content": assistant_answer},
            ],
        }
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
                 in a single batch.
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
        summarize_prompt = "You provide specific object identification in an image given a user's query. Don't answer the question itself but only provide the object name. Be concise in one sentence. If you are not sure, just reply 'i don't know'"
        
        inputs = []
        messages_batch = []
        for query, image in zip(queries, images):
            messages = [
                {"role": "system", "content": summarize_prompt},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Analyze the image and this question -- {query}. Identify the object name in the image for web search."}]},
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
        
        # Generate summaries in a single batch call
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=30,  # Short summary only
                skip_special_tokens=True
            )
        )
        
        # Extract and clean summaries
        summaries = [output.outputs[0].text.strip() for output in outputs]
        print(f"Generated {len(summaries)} image summaries: {summaries}")

        for session_id, answer, history in zip(session_ids, summaries, messages_batch): 
            self._log_for_sft(session_id, answer, history)

        return summaries
    
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
        
        # Retrieve relevant information for each query
        for i, search_query in enumerate(search_queries):
            results = self.search_pipeline(search_query, k=NUM_SEARCH_RESULTS)
            search_results_batch.append(results)
        
        # Prepare formatted inputs with RAG context for each query
        inputs = []
        messages_batch = []
        for idx, (query, image, message_history, search_results) in enumerate(
            zip(queries, images, message_histories, search_results_batch)
        ):
            # Create system prompt with RAG guidelines
            # SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions about the provided image."
            #                "Keep your response concise and to the point. If you don't know the answer, respond with 'I don't know'.")
            
            system_prompt = """You are in factual Q&A competition. Please respond concisely and truthfully in 65 words or less. If you don't know the answer, respond with 'I don't know'."""
            #user_prompt = """Context information is below. {context_str} Given the context information and using your prior knowledge, please provide your answer in concise style. End your answer with a period. Answer the question in one line only. Question: {query_str} Answer: """
            user_prompt = (
                "Given the context below and the image, answer the question truthfully in one line. "
                "Use context to support your answer explicitly. If insufficient information is available, say so.\n\n"
                "##Searched Context: {context_str}\n\n"
                "##Question: {query_str}\n\n"
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image"}]}
            ]
            
            # Add conversation history for multi-turn conversations
            if message_history:
                messages = messages + message_history
            
            # Add the current query
            messages.append({"role": "user", "content": user_prompt.format(context_str=rag_context, query_str=query)})
            
            # Apply chat template
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


        # Step 3: Generate responses using the batch of RAG-enhanced prompts
        print(f"Generating responses for {len(inputs)} queries")
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=MAX_GENERATION_TOKENS,
                skip_special_tokens=True
            )
        ) 

        # Extract and return the generated responses
        responses = [output.outputs[0].text for output in outputs]
        print(f"Successfully generated {len(responses)} responses")
        
        for session_id, answer, history in zip(session_ids, responses, messages_batch): 
            self._log_for_sft(session_id, answer, history)

        return inputs

    def batch_generate_response(
        self,
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
        
        # Step 1: Batch summarize all images for search terms
        image_summaries = self.batch_summarize_images(session_ids, queries, images)
        
        # Step 2: Prepare RAG-enhanced inputs in batch
        responses = self.prepare_rag_enhanced_inputs(session_ids, 
            queries, images, image_summaries, message_histories
        )
        
        return responses
