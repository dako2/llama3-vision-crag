#!/usr/bin/env python3
"""
Evaluator Script for CRAG-MM dataset

This script evaluates an agent (using a user-provided agent `UserAgent` as configured in `agents/user_config.py`) 
on the CRAG-MM dataset. It generates responses, evaluates them (using an optional semantic evaluation model via OpenAI API),
computes multi-turn conversation metrics, and optionally saves the results.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

# Set tokenizers parallelism before importing any HF libraries
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import pandas as pd
import tqdm
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

from agents.base_agent import BaseAgent
from crag_batch_iterator import CRAGTurnBatchIterator
from utils import display_results, ensure_crag_cache_dir_is_configured
from tokenizers import Tokenizer

from dotenv import load_dotenv
import os
load_dotenv()

from llm_logger import llm_logger

import pickle
import base64
from io import BytesIO
from PIL import Image

def encode_image_to_base64(image_obj):
    """Convert a PIL image or ndarray to base64 string."""
    if isinstance(image_obj, Image.Image):
        img = image_obj
    else:
        img = Image.fromarray(image_obj)

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

# Load environment variables
ensure_crag_cache_dir_is_configured()

console = Console()

# Constants for configuration
DEFAULT_EVAL_MODEL = "gpt-4.1"
MAX_API_RETRIES = 3
DEFAULT_NUM_WORKERS = 1

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 1

NUM_OF_CASE = 100

MAX_RESPONSE_LENGTH_IN_TOKENS = 75


class CRAGTurnEvaluationResult(BaseModel):
    """Structured output model for CRAG turn evaluation results."""
    accuracy: bool


def grok(messages):
    # Initialize the OpenAI client with xAI's API endpoint and key
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )
    # Make a request to the Grok API
    try:
        completion = client.chat.completions.create(
            model="grok-3-latest",  # Specify the Grok model (e.g., grok-beta)
            messages=messages,
        )
        # Print the response
        return completion.choices[0].message.parsed
    except Exception as e:
        return f"Error: {str(e)}"

class CRAGEvaluator:
    """
    A class to evaluate an agent on the CRAG-MM dataset.

    This evaluator generates responses, evaluates them (optionally using a semantic evaluation model),
    computes multi-turn conversation metrics, and (optionally) saves the results.
    """
    def __init__(
        self,
        dataset: Dataset,
        
        eval_model_name: str | None = None,
        num_conversations: int | None = None,
        show_progress: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        self.dataset = dataset
        
        self.eval_model_name = eval_model_name
        self.num_conversations = num_conversations
        self.show_progress = show_progress
        self.num_workers = num_workers

        # Internal state for evaluation; these are set during initialization
        self.batch_iterator: CRAGTurnBatchIterator | None = None
        self.conversations_count: int = 0
        self.agent_response_map: dict[str, str] = {}
        self.all_turn_data: list[dict[str, any]] = []
        self.session_ids_evaluated: set[str] = set()

    @staticmethod
    def get_system_message() -> str:
        """
        Returns the system message for the evaluator.
        """
        return (
            "You are an expert evaluator for question answering systems. "
            "Your task is to determine if a prediction correctly answers a question based on the ground truth.\n\n"
            "Rules:\n"
            "1. The prediction is correct if it captures all the key information from the ground truth.\n"
            "2. The prediction is correct even if phrased differently as long as the meaning is the same.\n"
            "3. The prediction is incorrect if it contains incorrect information or is missing essential details.\n"
            "Output a JSON object with a single field 'accuracy' whose value is true or false."
        )

    def attempt_api_call(
        self,
        client: OpenAI,
        model_name: str,
        messages: list,
        max_retries: int = MAX_API_RETRIES,
    ) -> CRAGTurnEvaluationResult | None:
        """
        Attempt a structured output call to the OpenAI API with retries.

        Args:
            client: The OpenAI client instance to use for the API call.
            model_name: The model to query (e.g., "gpt-4o-mini").
            messages: List of message objects for the conversation.
            max_retries: Maximum number of retry attempts before giving up.

        Returns:
            CRAGTurnEvaluationResult object if successful, None if all attempts fail.
        """
        for attempt in range(max_retries):
            try:
                completion = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    response_format=CRAGTurnEvaluationResult,
                )
                return completion.choices[0].message.parsed
            except Exception as e:
                error_message = f"API call failed on attempt {attempt + 1}/{max_retries}: {str(e)}"
                if attempt == max_retries - 1:
                    console.print(f"[red]Failed after {MAX_API_RETRIES} attempts: {str(e)}[/red]")
                else:
                    console.print(f"[yellow]{error_message}, retrying...[/yellow]")
        return None

    def evaluate_response(self, crag_turn_data: dict[str, any]) -> dict[str, any]:
        """
        Evaluate a single response and return evaluation results.

        Args:
            crag_turn_data: A dictionary containing query, ground truth, and agent response.

        Returns:
            A dictionary with evaluation results added to crag_turn_data.
        """
        agent_response = crag_turn_data["agent_response"]
        ground_truth = crag_turn_data["ground_truth"]
        query = crag_turn_data["query"]

        is_idk = "i don't know" in agent_response.lower()
        is_exact_match = agent_response.strip().lower() == ground_truth.strip().lower()
        is_semantically_correct = False
        api_response = None

        # Begin by assuming exact match correctness
        is_correct = is_exact_match

        # Use semantic evaluation if not an exact match and an evaluation model is provided.
        if not is_idk and not is_exact_match and self.eval_model_name:
            local_openai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                #base_url="https://api.x.ai/v1",
            )
            messages = [
                {"role": "system", "content": self.get_system_message()},
                {"role": "user", "content": f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {agent_response}\n"},
            ]
            api_response = self.attempt_api_call(local_openai_client, self.eval_model_name, messages)
            #api_response = grok(messages)

            if api_response:
                is_semantically_correct = api_response.accuracy
                is_correct = is_semantically_correct
        if is_exact_match:
            is_semantically_correct = True

        return {
            **crag_turn_data,
            "is_exact_match": is_exact_match,
            "is_correct": is_correct,
            "is_miss": is_idk,
            "is_semantically_correct": is_semantically_correct,
            "api_response": api_response.model_dump() if api_response else None,
        }

    def initialize_evaluation(self) -> None:
        """
        Initialize variables needed for agent evaluation.

        This method sets internal state including the batch iterator, conversation count, 
        agent response map, and turn data list.
        """
        console.print(f"[blue]Starting evaluation with {self.num_workers} workers[/blue]")
        if self.eval_model_name:
            console.print(f"[blue]Using semantic evaluation with model: {self.eval_model_name}[/blue]")

        self.conversations_count = len(self.dataset) if self.num_conversations is None else min(self.num_conversations, len(self.dataset))
        batch_size = 8
        self.agent_response_map = {}
        self.all_turn_data = []
        self.session_ids_evaluated = set()

        # Instantiate the CRAG turn based batch iterator 
        self.batch_iterator = CRAGTurnBatchIterator(dataset=self.dataset, batch_size=batch_size, shuffle=False)

    def generate_agent_responses(self, progress_callback: Callable[[int, int], None] = None) -> None:
        """
        Phase 1: Generate agent responses for each turn in the dataset.
        Phase 1: Generate agent responses for each turn in the dataset.

        This method iterates over the dataset batches using the internal batch iterator and updates the evaluator's state
        with agent responses and turn data.
        """
        if self.batch_iterator is None:
            raise ValueError("Batch iterator is not initialized. Please call initialize_evaluation() first.")

        for batch_idx, batch in enumerate(tqdm.tqdm(self.batch_iterator, desc="Generating responses", disable=not self.show_progress)):
            session_ids = batch["session_ids"]

            # Generate responses for the current batch
            agent_responses, messages_batch = lookup_results(session_ids)
            
            # Collect responses and add evaluation data
            for idx, interaction_id in enumerate(interaction_ids):
                agent_response = agent_responses[idx]
                self.agent_response_map[interaction_id] = agent_response
                self.all_turn_data.append({
                    "session_id": batch["session_ids"][idx],
                    "interaction_id": interaction_id,
                    "turn_idx": batch["turn_idxs"][idx],
                    "is_ego": batch["image_urls"][idx] is None,
                    "image_quality": batch["image_qualities"][idx],
                    "query_category": batch["query_categories"][idx],
                    "domain": batch["domains"][idx],
                    "dynamism": batch["dynamisms"][idx],
                    "query": queries[idx],
                    "ground_truth": batch["answers"][idx],
                    "agent_response": agent_response,
                    "messages_batch": messages_batch,
                    "total_turn_count": batch["total_turn_counts"][idx],
                    "interaction_id_history": interaction_id_histories[idx],
                    "image": images[idx]
                })
                self.session_ids_evaluated.add(batch["session_ids"][idx])

            if progress_callback:
                conversations_evaluated = len(self.session_ids_evaluated)
                progress_callback(conversations_evaluated, self.conversations_count)

            if len(self.session_ids_evaluated) > self.conversations_count:
                console.print(f"[yellow]Already evaluated {len(self.session_ids_evaluated)} conversations. Abruptly stopping evaluation.[/yellow]")
                break

    def evaluate_agent_responses(
        self,
        turn_data: list[dict[str, any]],
        progress_callback: Callable[[int, int], None] = None
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
        """
        Phase 2: Evaluate agent responses and calculate scores.

        This method uses a thread-based parallel executor to avoid pickling issues.
        Args:
            turn_data: List of turn data including agent responses.
        Returns:
            A tuple containing turn evaluation results and score dictionaries.
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.evaluate_response, data) for data in turn_data]
            for future_idx, future in tqdm.tqdm(enumerate(as_completed(futures)), total=len(futures), desc="Evaluating responses", disable=not self.show_progress):
                results.append(future.result())
                if progress_callback is not None:
                    progress_callback(future_idx, len(turn_data))

        # Convert the interim evaluation results to a pandas dataframe
        turn_evaluation_results_df = pd.DataFrame(results)
        turn_evaluation_results_df = turn_evaluation_results_df.sort_values(by=["session_id", "turn_idx"])

        ego_turn_evaluation_results_df = turn_evaluation_results_df[turn_evaluation_results_df["is_ego"] == True]

        all_scores_dictionary = self.calculate_scores(turn_evaluation_results_df)
        ego_scores_dictionary = self.calculate_scores(ego_turn_evaluation_results_df)

        turn_evaluation_results = {"all": turn_evaluation_results_df, "ego": ego_turn_evaluation_results_df}
        score_dictionaries = {"all": all_scores_dictionary, "ego": ego_scores_dictionary}

        return turn_evaluation_results, score_dictionaries, results

    def calculate_scores(self, turn_evaluation_results_df: pd.DataFrame) -> dict[str, float]:
        """
        Calculate scores for both single-turn and multi-turn conversations.

        Args:
            turn_evaluation_results_df: DataFrame with evaluation results for turns.
        Returns:
            Dictionary of calculated metrics.
        """
        multi_turn_conversation_score_map: dict[str, float] = {}

        def _set_is_correct_false_after_consecutive(group: pd.DataFrame) -> pd.DataFrame:
            """
            Mark as is_miss after consecutive incorrect responses
            and calculate multi-turn conversation score for each conversation.
            """
            group_copy = group.copy().reset_index(drop=True)
            for i in range(1, len(group_copy)):
                if not group_copy.loc[i - 1, 'is_correct'] and not group_copy.loc[i, 'is_correct']:
                    group_copy.loc[i + 1:, 'is_correct'] = False
                    group_copy.loc[i + 1:, 'is_exact_match'] = False
                    group_copy.loc[i + 1:, 'is_miss'] = True
                    group_copy.loc[i + 1:, 'is_semantically_correct'] = False
                    break

            group_copy["is_hallucination"] = ~group_copy["is_correct"] & ~group_copy["is_miss"]
            multi_turn_conversation_score = group_copy["is_correct"].mean() - group_copy["is_hallucination"].mean()
            group_copy["multi_turn_conversation_score"] = multi_turn_conversation_score
            session_id = group_copy.iloc[0]["session_id"]
            multi_turn_conversation_score_map[session_id] = multi_turn_conversation_score
            return group_copy

        turn_evaluation_results_df = turn_evaluation_results_df.groupby("session_id", group_keys=False)[turn_evaluation_results_df.columns].apply(_set_is_correct_false_after_consecutive)

        total = len(turn_evaluation_results_df)
        correct_exact = turn_evaluation_results_df["is_exact_match"].sum()
        correct = turn_evaluation_results_df["is_correct"].sum()
        miss = turn_evaluation_results_df["is_miss"].sum()
        hallucination = total - (correct + miss)

        exact_match = correct_exact / total
        accuracy = correct / total
        missing = miss / total
        hallucination_rate = hallucination / total
        truthfulness_score = ((2 * correct + miss) / total) - 1 if total > 1 else 0.0
        mean_multi_turn_conversation_score = np.mean(list(multi_turn_conversation_score_map.values()))

        scores_dictionary = {
            "total": float(total),
            "correct_exact": float(correct_exact),
            "correct": float(correct),
            "miss": float(miss),
            "hallucination": float(hallucination),
            "exact_match": float(exact_match),
            "accuracy": float(accuracy),
            "missing": float(missing),
            "hallucination_rate": float(hallucination_rate),
            "truthfulness_score": float(truthfulness_score),
            "mean_multi_turn_conversation_score": float(mean_multi_turn_conversation_score)
        }

        return scores_dictionary

    def posteval_and_log_for_sft(self, eval_results, output_path="sft_dataset.pkl"):
        """Process eval results and save to pickle file for fine-tuning."""
        all_sft_data = []

        for data in eval_results:
            session_id = data.get("session_id")
            print("Processing:", session_id)

            # Determine final assistant response
            if data.get("is_semantically_correct") or data.get("is_correct") or data.get("is_miss"):
                assistant_text = data["ground_truth"]
            else:
                assistant_text = "i don't know"

            messages = data["messages_batch"][0]
            system_prompt = messages[0]["content"]
            user_text = messages[2]["content"]

            # Encode image safely
            try:
                image_b64 = encode_image_to_base64(data["image"])
            except Exception as e:
                print(f"Skipping {session_id} due to image error: {e}")
                continue

            new_format = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_b64},
                            {"type": "text", "text": system_prompt + "\n" + user_text}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": assistant_text}
                        ]
                    }
                ]
            }

            all_sft_data.append(new_format)

        # Save to pickle
        with open(output_path, "wb") as f:
            pickle.dump(all_sft_data, f)

        print(f"✅ Saved {len(all_sft_data)} examples to {output_path}")
        return "successfully saved sft data"
        
    def save_results(self, turn_evaluation_results: dict[str, any], scores_dictionary: dict[str, any], output_dir: str) -> None:
        """
        Save evaluation results to the specified directory.

        Args:
            turn_evaluation_results: The evaluation results to save.
            scores_dictionary: The scores dictionary to save.
            output_dir: Path where to save the results.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)
        turn_evaluation_results["all"].to_csv(os.path.join(output_dir, "turn_evaluation_results_all.csv"), index=False)
        turn_evaluation_results["ego"].to_csv(os.path.join(output_dir, "turn_evaluation_results_ego.csv"), index=False)
        with open(os.path.join(output_dir, "scores_dictionary.json"), "w") as f:
            json.dump(scores_dictionary, f, indent=2)

    def evaluate_agent(self) -> tuple[dict[str, any], dict[str, any]]:
        """
        Evaluate an agent on a dataset and return performance metrics.

        Returns:
            A tuple containing a dictionary of turn evaluation results and a dictionary of scores.
        """
        # Phase 0: Initialize evaluation state
        self.initialize_evaluation()
        
        # Phase 1: Generate agent responses (updates internal state)
        def _generation_progress_callback(conversations_evaluated: int, total_conversations: int) -> None:
            # Can be useful to track progress of the evaluation
            # console.log(f"[blue]Generated responses for {conversations_evaluated}/{total_conversations} conversations[/blue]")
            pass
            
        self.generate_agent_responses(_generation_progress_callback)
        
        # Phase 2: Evaluate responses using stored turn data
        
        def _evaluation_progress_callback(turn_evaluated: int, total_turns: int) -> None:
            # Can be useful to track progress of the evaluation
            # console.log(f"[blue]Evaluated {turn_evaluated}/{total_turns} turns[/blue]")
            pass
            
        turn_evaluation_results, score_dictionaries, eval_results_dict = self.evaluate_agent_responses(self.all_turn_data, _evaluation_progress_callback)

        #self.posteval_and_log_for_sft(eval_results_dict)

        return turn_evaluation_results, score_dictionaries
    
    def truncate_agent_responses(self, agent_responses: list[str]) -> list[str]:
        """
        Truncate each agent response to the maximum allowed length.
        """
        encodings = self.tokenizer.encode_batch(agent_responses)
        trimmed_agent_responses = [self.tokenizer.decode(enc.ids) for enc in encodings]
        return trimmed_agent_responses    

import json

# Let's say your JSONL has been loaded into a list called 'all_sessions'
with open("results/sft_response_data_case_5_web_search_rephrase.jsonl") as f:
    all_sessions = [json.loads(line) for line in f]

# Build the lookup dictionary
lookup_dict = {}
for entry in all_sessions:
    session_id = entry["session_id"]
    # Assume each entry["messages"][-1]["role"] == "assistant"
    assistant_message = entry["messages"][-1]["content"]
    lookup_dict[session_id] = assistant_message
    
def lookup_results(session_ids):
    """
    Given a list of session_ids, return (agent_responses, messages_batch).
    agent_responses: list of assistant responses.
    messages_batch:   list of message dicts from each session (can also just return None if not needed).
    """
    agent_responses = []
    messages_batch = []

    for sid in session_ids:
        # Handle missing session_id
        response = lookup_dict.get(sid, "i don't know")
        agent_responses.append(response)
        # Optionally return the entire message batch if you want (or just None)
        # Here we just return None as a placeholder
        messages_batch.append(None)
    return agent_responses, messages_batch

dataset = load_dataset(
    "crag-mm-2025/crag-mm-single-turn-public",
    split="validation",
    streaming=False,              # set True if RAM is tight
)

evaluator = CRAGEvaluator(
    dataset=dataset,
   
    eval_model_name="gpt-4.1",
    num_conversations=-1,
    show_progress=True,
    num_workers=8,
)

turn_evaluation_results, score_dictionaries = evaluator.evaluate_agent()
display_results(
    console,
    turn_evaluation_results["all"],
    score_dictionaries["all"],
    display_conversations=100,
    is_ego=False,
    is_multi_turn=False,
)
 
