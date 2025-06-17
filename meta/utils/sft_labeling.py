#!/usr/bin/env python
"""postprocess_sft_labeling.py – debug‑friendly edition
──────────────────────────────────────────────────────────────────────────────
Reads a CRAG‑MM evaluation CSV and writes SFT‑ready chat JSONL.

This version **removes the extra `main()` wrapper** so you can set breakpoints
and step through the script top‑to‑bottom more easily in a debugger.

Defaults:
  • Input  → ./temp/turn_evaluation_results_all.csv
  • Output → ./sft_data.jsonl
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

###############################################################################
# Logging setup
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOG = logging.getLogger("postprocess_sft_labeling")

###############################################################################
# Helpers
###############################################################################

def _iter_csv(path: Path) -> Generator[Dict[str, Any], None, None]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _safe_json_parse(raw: str) -> Any:
    """Parse JSON or single‑quoted pseudo JSON. Unwrap double‑encoded strings."""
    if not raw:
        raise ValueError("Empty field")
    # If the entire cell is quoted again ("…") try a de‑quote first
    if raw.startswith("\"") and raw.endswith("\""):
        try:
            raw = json.loads(raw)  # remove one level
        except Exception:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def _norm_content(items: List[Any]) -> List[Dict[str, Any]]:
    canon: List[Dict[str, Any]] = []
    for itm in items:
        if isinstance(itm, dict):
            canon.append(itm)
        elif isinstance(itm, str):
            canon.append({"type": "text", "text": itm})
        else:
            canon.append({"type": "text", "text": str(itm)})
    return canon


def _normalize_messages(obj: Any) -> List[Dict[str, Any]]:
    """Return canonical list[dict] for *messages*.

    Handles:
      • list / dict / str at top level
      • dicts whose *content* may be list **or** str
    """
    if obj is None:
        return []

    # Case A: already a list of message objects / primitives
    if isinstance(obj, list):
        canon: List[Dict[str, Any]] = []
        for itm in obj:
            if isinstance(itm, dict):
                content = itm.get("content")
                if isinstance(content, list):
                    itm["content"] = _norm_content(content)
                elif isinstance(content, str):
                    itm["content"] = _norm_content([content])
                canon.append(itm)
            else:  # primitive – wrap as user message
                canon.append({"role": "user", "content": _norm_content([itm])})
        return canon

    # Case B: single dict
    if isinstance(obj, dict):
        content = obj.get("content")
        if isinstance(content, list):
            obj["content"] = _norm_content(content)
        elif isinstance(content, str):
            obj["content"] = _norm_content([content])
        return [obj]

    # Case C: primitive
    if isinstance(obj, str):
        return [{"role": "user", "content": _norm_content([obj])}]

    raise TypeError(f"Unsupported messages type: {type(obj)}")

def _extract_instruction_and_image(rec: Dict[str, Any]) -> tuple[str, Optional[str]]:
    raw = rec.get("messages", "")
    try:
        msgs = _normalize_messages(_safe_json_parse(raw))
    except Exception:
        msgs = []

    instruction: Optional[str] = None
    image_id: Optional[str] = None

    for msg in msgs:
        if msg.get("role") != "user":
            continue
        for part in msg.get("content", []):
            if isinstance(part, dict):
                if part.get("type") == "text" and not instruction:
                    instruction = str(part.get("text", "")).strip()
                if part.get("type") == "image" and not image_id:
                    image_id = part.get("image")
            elif isinstance(part, str) and not instruction:
                instruction = part.strip()
        if instruction:
            break

    # Fallbacks
    if not instruction:
        instruction = str(rec.get("query", "")).strip()
    if not instruction:
        raise ValueError("No instruction text located.")

    if not image_id:
        image_id = rec.get("image") or rec.get("session_id")

    return instruction, image_id


def _make_answer(rec: Dict[str, Any]) -> str:
    return str(rec.get("ground_truth", "")) if str(rec.get("is_correct")).lower() in {"true", "1", "yes"} else "I don't know."


def _to_chat(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    instruction, image_id = _extract_instruction_and_image(rec)
    answer = _make_answer(rec)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image_id},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]

###############################################################################
# Top‑level script (no extra main() wrapper) – easy to set breakpoints here
###############################################################################
DEFAULT_IN = "./turn_evaluation_results_all.csv"
DEFAULT_OUT = "./sft_data.jsonl"

parser = argparse.ArgumentParser(description="CRAG‑MM → SFT chat converter (debug‑friendly)")
parser.add_argument("--input", default=DEFAULT_IN, help="CSV with evaluation flags & messages")
parser.add_argument("--output", default=DEFAULT_OUT, help="Output JSONL file")
args = parser.parse_args()

INPUT_PATH = Path(args.input).expanduser()
OUTPUT_PATH = Path(args.output).expanduser()

LOG.info("🔍 Reading   %s", INPUT_PATH)
LOG.info("💾 Writing → %s", OUTPUT_PATH)

total = ok = skipped = 0

with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
    for rec in _iter_csv(INPUT_PATH):
        total += 1
        try:
            chat = _to_chat(rec)  # ← set a breakpoint here to inspect per‑row

            fout.write(json.dumps({"messages":chat}, ensure_ascii=False) + "\n")
            ok += 1
        except Exception as exc:
            LOG.error("🚫  Skipping row %d: %s", total, exc)
            skipped += 1

LOG.info("✅ Finished. kept=%d | skipped=%d | total=%d", ok, skipped, total)
