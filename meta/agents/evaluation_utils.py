# evaluation_utils.py
"""Reusable evaluation helpers extracted from the original CRAG-MM evaluator.

This module contains fully self‑contained, framework‑agnostic utilities so you can
plug them into *any* project that needs to grade a list of predictions against
references and compute CRAG‑style metrics (exact match, accuracy, miss /
hallucination rates, truthfulness, multi‑turn score …).

Typical usage
-------------

>>> import pandas as pd, evaluation_utils as ev
>>> rows = [
...     {"session_id": "s1", "query": "Who wrote Hamlet?",
...      "ground_truth": "William Shakespeare",
...      "prediction": "William Shakespeare"},
... ]
>>> df = pd.DataFrame(rows)
>>> df = ev.evaluate_dataframe(df)          # adds boolean columns
>>> df = ev.add_finetune_answer(df)        # new helper
>>> scores = ev.calculate_scores(df)
>>> print(scores["accuracy"], df.head())

If you would like an LLM‑as‑a‑judge step (semantic correctness) just pass the
model name (e.g. "gpt‑4o-mini"). All OpenAI calls are automatically retried up
to three times.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console
import json
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MAX_API_RETRIES = 3
console = Console()  # rich console for coloured logging (optional)

# ---------------------------------------------------------------------------
# Pydantic schema for structured OpenAI responses (if you use LLM‑judge)
# ---------------------------------------------------------------------------

class _CRAGTurnEvaluationResult(BaseModel):
    """Schema expected from the OpenAI ``response_format`` call."""

    accuracy: bool


# ---------------------------------------------------------------------------
# Low‑level helpers
# ---------------------------------------------------------------------------

def save_dataframe_to_jsonl(df: pd.DataFrame, path: str, *, append: bool = True):
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\\n")

def _get_system_message() -> str:
    """System prompt used when querying an LLM to act as semantic judge."""

    return (
        "You are an expert evaluator for question answering systems. "
        "Your task is to determine if a prediction correctly answers a question based on the ground truth.\n\n"
        "Rules:\n"
        "1. The prediction is correct if it captures all the key information from the ground truth.\n"
        "2. The prediction is correct even if phrased differently as long as the meaning is the same.\n"
        "3. The prediction is incorrect if it contains incorrect information or is missing essential details.\n\n"
        "Output a JSON object with a single field 'accuracy' whose value is true or false."
    )


def _attempt_api_call(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    max_retries: int = _DEFAULT_MAX_API_RETRIES,
) -> Optional[_CRAGTurnEvaluationResult]:
    """Wrapper around ``client.beta.chat.completions.parse`` with retries."""

    for attempt in range(max_retries):
        try:
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=_CRAGTurnEvaluationResult,
            )
            return completion.choices[0].message.parsed  # type: ignore[attr-defined]
        except Exception as exc:  # pylint: disable=broad-except
            if attempt == max_retries - 1:
                console.print(f"[red]OpenAI call failed: {exc}[/red]")
            else:
                console.print(
                    f"[yellow]OpenAI call failed (attempt {attempt + 1}/{max_retries}), retrying …[/yellow]"
                )
    return None


# ---------------------------------------------------------------------------
# Public API — single‑turn evaluation
# ---------------------------------------------------------------------------

def evaluate_response(
    query: str,
    ground_truth: str,
    prediction: str,
    *,
    eval_model_name: Optional[str] = None,
    openai_client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """Evaluate a **single** prediction and return a rich dict with flags."""

    pred_clean = prediction.strip()
    gt_clean = ground_truth.strip()

    is_idk = "i don't know" in pred_clean.lower()
    is_exact_match = pred_clean.lower() == gt_clean.lower()

    is_semantically_correct = False
    is_correct = is_exact_match  # may update below
    api_response: Optional[_CRAGTurnEvaluationResult] = None

    if not (is_idk or is_exact_match) and eval_model_name is not None:
        client = openai_client or OpenAI()
        messages = [
            {"role": "system", "content": _get_system_message()},
            {
                "role": "user",
                "content": (
                    f"Question: {query}\nGround truth: {ground_truth}\n"
                    f"Prediction: {prediction}\n"
                ),
            },
        ]
        api_response = _attempt_api_call(client, eval_model_name, messages)
        if api_response is not None:
            is_semantically_correct = api_response.accuracy
            is_correct = is_semantically_correct

    if is_exact_match:
        is_semantically_correct = True

    return {
        "query": query,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "is_exact_match": is_exact_match,
        "is_correct": is_correct,
        "is_miss": is_idk,
        "is_semantically_correct": is_semantically_correct,
        "api_response": api_response.model_dump() if api_response else None,
    }


# ---------------------------------------------------------------------------
# Batch helpers / metrics
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = {
    "session_id",
    "is_correct",
    "is_exact_match",
    "is_miss",
}


def calculate_scores(turns: pd.DataFrame) -> Dict[str, float]:
    """Compute corpus‑level CRAG metrics from an evaluated DataFrame."""

    missing = _REQUIRED_COLUMNS - set(turns.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    df = turns.copy().reset_index(drop=True)
    conversation_scores: Dict[str, float] = {}

    def _apply_multi_turn(group: pd.DataFrame) -> pd.DataFrame:  # noqa: N802
        g = group.copy().reset_index(drop=True)
        for i in range(1, len(g)):
            if not g.loc[i - 1, "is_correct"] and not g.loc[i, "is_correct"]:
                g.loc[i + 1 :, ["is_correct", "is_exact_match"]] = False
                g.loc[i + 1 :, "is_miss"] = True
                break
        g["is_hallucination"] = ~g["is_correct"] & ~g["is_miss"]
        score = g["is_correct"].mean() - g["is_hallucination"].mean()
        conversation_scores[g.iloc[0]["session_id"]] = score
        g["multi_turn_conversation_score"] = score
        return g

    df = df.groupby("session_id", group_keys=False).apply(_apply_multi_turn)

    total = len(df)
    correct_exact = int(df["is_exact_match"].sum())
    correct = int(df["is_correct"].sum())
    miss = int(df["is_miss"].sum())
    hallucination = total - (correct + miss)

    exact_match = correct_exact / total if total else 0.0
    accuracy = correct / total if total else 0.0
    missing = miss / total if total else 0.0
    hallucination_rate = hallucination / total if total else 0.0
    truthfulness_score = ((2 * correct + miss) / total) - 1 if total > 1 else 0.0
    mean_multi_turn_conversation_score = float(np.mean(list(conversation_scores.values()))) if conversation_scores else 0.0

    return {
        "total": float(total),
        "correct_exact": float(correct_exact),
        "correct": float(correct),
        "miss": float(miss),
        "hallucination": float(hallucination),
        "exact_match": exact_match,
        "accuracy": accuracy,
        "missing": missing,
        "hallucination_rate": hallucination_rate,
        "truthfulness_score": truthfulness_score,
        "mean_multi_turn_conversation_score": mean_multi_turn_conversation_score,
    }


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def evaluate_dataframe(
    df: pd.DataFrame,
    *,
    query_col: str = "query",
    gt_col: str = "ground_truth",
    pred_col: str = "prediction",
    eval_model_name: Optional[str] = None,
    openai_client: Optional[OpenAI] = None,
) -> pd.DataFrame:
    """Vectorised wrapper that appends evaluation flags to *df*."""

    records: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        records.append(
            evaluate_response(
                getattr(row, query_col),
                getattr(row, gt_col),
                getattr(row, pred_col),
                eval_model_name=eval_model_name,
                openai_client=openai_client,
            )
        )

    eval_df = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), eval_df], axis=1)


# ---------------------------------------------------------------------------
# New helper: build finetune answer column
# ---------------------------------------------------------------------------

def add_finetune_answer(
    df: pd.DataFrame,
    *,
    ground_truth_col: str = "ground_truth",
    output_col: str = "finetune_answer",
) -> pd.DataFrame:
    """Attach a *finetune_answer* column ready for SFT / RLHF pipelines.

    Logic:
    - If the model answered correctly (exact or semantic) → copy *ground_truth*.
    - Otherwise (miss or hallucination) → "I don't know".

    Requires columns ``is_correct`` and ``is_miss`` that are created by
    :func:`evaluate_dataframe`.
    """

    if {"is_correct", "is_miss"} - set(df.columns):
        raise ValueError(
            "You must call `evaluate_dataframe` before `add_finetune_answer` so "
            "that 'is_correct' and 'is_miss' columns exist."
        )

    df = df.copy()
    df[output_col] = np.where(
        df["is_correct"],
        df[ground_truth_col].astype(str),  # ensures 1D string output
        "I don't know"
    )

    return df


__all__ = [
    "evaluate_response",
    "evaluate_dataframe",
    "calculate_scores",
    "add_finetune_answer",
]
