from typing import Dict, List, Any
import os
import time
import torch
from PIL import Image
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

from crag_web_result_fetcher import WebSearchResult
import vllm
from agents._reranker import SentenceReranker

from pathlib import Path
import pandas as pd
import agents.evaluation_utils as ev

fast_rr = SentenceReranker()
# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 8

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


# GPU utilization settings 
# Change VLLM_TENSOR_PARALLEL_SIZE during local runs based on your available GPUs
# For example, if you have 2 GPUs on the server, set VLLM_TENSOR_PARALLEL_SIZE=2. 
# You may need to uncomment the following line to perform local evaluation with VLLM_TENSOR_PARALLEL_SIZE>1. 

#### Please ensure that when you submit, VLLM_TENSOR_PARALLEL_SIZE=1. 
#os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.95 

# These are model specific parameters to get the model to run on a single NVIDIA L40s GPU
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2
MAX_GENERATION_TOKENS = 75

# Number of search results to retrieve
NUM_SEARCH_RESULTS = 5

def normalize_answer(text: str) -> str:
    """
    Collapse any answer that indicates uncertainty into
    the canonical string "i don't know".
    """
    text_lower = text.lower()
    uncertain_phrases = ["don't know", "don't", "not sure", "unable", "not", "not able to"]

    if any(phrase in text_lower for phrase in uncertain_phrases):
        return "I Don't Know"
    return text

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
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", #meta-llama/Llama-3.2-11B-Vision-Instruct
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
        self.timestamp = int(time.time())
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
                
        if self.tokenizer.chat_template is None:
            tmpl_file = Path(self.model_name) / "chat_template.json"
            if tmpl_file.exists():
                self.tokenizer.chat_template = tmpl_file.read_text()
                print("Chat template loaded from local file.")
            else:
                # fallback – inline template for Llama-3.2 Vision
                self.tokenizer.chat_template = (
                    "{% if messages[0]['role'] == 'system' %}"
                    "{{ messages[0]['content'] }}{% endif %}"
                    "{% for m in messages[1:] %}"
                    "{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' }}"
                    "{% endfor %}"
                    "{{ '<|im_start|>assistant\\n' }}"
                )
                print("Injected default chat template.")

        print("Models loaded successfully")

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
    
    def batch_summarize_images(self, queries, images: List[Image.Image]) -> List[str]:
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
        summarize_prompt = """Provide the exact identity name that I was asking previously in the image -- {query}. If you are not sure, please respond 'I don't know' directly."""
        #summarize_prompt = """Identity the specific name of the object that the user is asking in the image. Don't answer the question itself but provide only the object identification that the user is asking {query}.If """
        
        inputs = []
        messages_batch = []
        for query, image in zip(queries, images):
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": summarize_prompt.format(query=query)}]},
            ]
            
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
            messages_batch.append(messages)
        
        # Generate summaries in a single batch call
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.01,
                top_p=0.9,
                max_tokens=50,  # Short summary only
                skip_special_tokens=True
            )
        )
        
        # Extract and clean summaries
        summaries = [normalize_answer(output.outputs[0].text.strip()) for output in outputs]
        print(f"Generated {len(summaries)} image summaries")

        return summaries, messages_batch

    def prepare_image_search_inputs(
        self,
        queries: List[str], 
        images: List[Image.Image], 
        image_summaries: List[str],
    ) -> List[dict]:
        pass
    
    def prepare_rag_enhanced_inputs(
        self, 
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
    
        # Retrieve relevant information for each query
        for query, summary in zip(queries, image_summaries):

            if summary == "i don't know":
                search_results_batch.append("")
                continue

            print("searching:",query)

            q = f"{query}...{summary}"

            rag_context = []
            results = self.search_pipeline(q, k=NUM_SEARCH_RESULTS)
            for i, result in enumerate(results):
                #result = WebSearchResult(result)

                snippet = result.get('page_snippet', '')
                print(result)
                rag_context.append(str(snippet))

            # # Add retrieved context if available
            # rag_context = ""
            # if search_results:
            #     #rag_context = "Here is some additional information that may help you answer:\n\n"
            #     for i, result in enumerate(search_results):
            #         # WebSearchResult is a helper class to get the full page content of a web search result.
            #         #
            #         # It first checks if the page content is already available in the cache. If not, it fetches  
            #         # the full page content and caches it.
            #         #
            #         # WebSearchResult adds `page_content` attribute to the result dictionary where the page 
            #         # content is stored. You can use it like a regular dictionary to fetch other attributes.
            #         #
            #         # result["page_content"] for complete page content, this is available only via WebSearchResult
            #         # result["page_url"] for page URL
            #         # result["page_name"] for page title
            #         # result["page_snippet"] for page snippet
            #         # result["score"] relavancy with the search query
            #         #print(result["page_content"])
            #         result = WebSearchResult(result)
            #         snippet = result.get('page_snippet', '')
            #         if snippet:
            #             rag_context += f"[Info {i+1}] {snippet}\n\n"

            t0 = time.time()
            new_results = fast_rr.batch_rerank(rag_context, q, k=5, min_score=0)
            print(new_results)
            
            flat_sentences = [s for group in new_results for (s, _) in group]

            print("Fast:", " ".join(flat_sentences))
            print("reranking takes %.1f seconds"%(time.time()-t0))
            search_results_batch.append(flat_sentences)
        
        # Prepare formatted inputs with RAG context for each query
        messages_batch = []
        for idx, (query, image, caption, message_history, rag_context) in enumerate(
            zip(queries, images, image_summaries, message_histories, search_results_batch)
        ):
            if caption == "i don't know":
                messages_batch.append([])
                continue

            #SYSTEM_PROMPT = """You are in factual Q&A competition. Please respond concisely and truthfully in 65 words or less. If you don't know the answer, respond with 'I don't know'."""
            #user_prompt = """Context information is below. {context_str} Given the context information and using your prior knowledge, please provide your answer in concise style. End your answer with a period. Answer the question in one line only. Question: {query_str} Answer: """
            user_prompt = (
                #"You are in factual Q&A competition. Please respond concisely and truthfully in 65 words or less. If you don't know the answer, respond with 'I don't know'."
                "You are a factual and knowledgable Q&A expert that answer the question about something truthfully in one line. "
                "Use context to support your answer explicitly. If insufficient information is available, say so.\n\n"
                "The image {caption}\n"
                "Some reference might be useful: {context_str}\n"
                "Based on the above information, answer my question: {query_str}\n"
                #"if you are not sure, please answer 'i don't know' directly."
            )

            # Structure messages with image and RAG context
            messages = [
                #{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}]}
            ]
            
            # Add conversation history for multi-turn conversations
            if message_history:
                messages = messages + message_history
                
            # Add the current query
            messages.append({"role": "user", "content": user_prompt.format(caption=caption,query_str=query, context_str=rag_context)})
            messages_batch.append(messages)
        
        return messages_batch

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

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
        session_ids: List[str],
        ground_truths: List[List[str]],
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
        
        images = resize_images(images)

        # Step 1: Batch summarize all images for search terms
        image_summaries, caption_messages_batch = self.batch_summarize_images(queries, images)

        # Step 2: Determine which queries should skip LLM generation
        should_skip = [summary.lower().strip() == "i don't know" for summary in image_summaries]

        #here the returned image_summaries might contain i don't know

        # Step 3: Prepare RAG-enhanced inputs only for non-skipped
        rag_inputs = []
        original_indices = []
        # Maintain message history for all queries (skipped and not skipped)
        full_messages_batch = [None] * len(queries)
        # Filter inputs for generation
        for idx, (skip, query, image, summary, history) in enumerate(zip(
            should_skip, queries, images, image_summaries, message_histories
        )):
            if skip:
                continue

            messages = self.prepare_rag_enhanced_inputs(
                [query], [image], [summary], [history]
            )[0]  # unpack single result

            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )

            rag_inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": image}
            })
            original_indices.append(idx)
            full_messages_batch[idx] = messages  # ✅ Save message to aligned batch

        # Step 4: Generate responses
        print(f"Generating responses for {len(rag_inputs)} queries")
        generated_outputs = self.llm.generate(
            rag_inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.5,
                top_p=0.9,
                max_tokens=MAX_GENERATION_TOKENS,
                skip_special_tokens=True
            )
        )
        generated_texts = [output.outputs[0].text for output in generated_outputs]
        print(f"Successfully generated {len(generated_texts)} responses")

        # Step 5: Merge skipped + generated back in original order
        predictions = [""] * len(queries)
        for idx, text in zip(original_indices, generated_texts):
            predictions[idx] = text
        for idx, skip in enumerate(should_skip):
            if skip:
                predictions[idx] = "I don't know"

        print(f"Successfully generated responses: {predictions} ")

        # rows = []
        # for sid, q, gt, pred, caption, mes in zip(session_ids, queries, ground_truths, predictions, image_summaries, full_messages_batch):
        #     rows.append({"session_id": sid, "turn_idx": 0, 
        #     "query": q,
        #     "ground_truth": gt[0],
        #     "prediction": pred,
        #     "caption": caption,
        #     "messages": mes,
        #     },)

        # # after predictions
        # df = pd.DataFrame(rows)
        # df = ev.evaluate_dataframe(df)           # flags
        
        # df = ev.add_finetune_answer(df)          # finetune_answer col
        
        # scores = ev.calculate_scores(df)

        # print("Accuracy:", scores["accuracy"])

        # ev.save_dataframe_to_jsonl(df, "./data/finetune_data_%d.jsonl"%(self.timestamp), append=True)

        # final_output = []
        # for p, c in zip(predictions, image_summaries):
        #     final_output.append(f"{c} | {p}")

        return predictions, full_messages_batch, image_summaries, caption_messages_batch
