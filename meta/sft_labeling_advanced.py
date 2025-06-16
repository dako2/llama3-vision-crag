#!/usr/bin/env python
"""postprocess_sft_labeling.py – debug‑friendly, two answer modes
──────────────────────────────────────────────────────────────────────────────
Convert CRAG‑MM evaluation CSV → SFT‑ready chat JSONL.

Two answer‑generation strategies are now available:

* **simple**  – classic rule: copy `ground_truth` when `is_correct`, else "I don't know."
* **context** – richer logic: if row is wrong **but** information is sufficient
  (`if_-1_is_info_enough_for_truth == 1` or Grok says so) we still emit the
  correct answer prefixed with a short cue.  Otherwise same as *simple*.

Choose via `--answer_mode simple|context` (default **context**).  Combine with
`--check_sufficiency` to call Grok for rows lacking the flag.

python sft_labeling_advanced.py --answer_mode context --check_sufficiency

"""
from __future__ import annotations

import argparse, ast, csv, json, logging, os, requests
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

###############################################################################
# Logging setup
###############################################################################
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("postprocess_sft_labeling")

###############################################################################
# Grok sufficiency check (optional)
###############################################################################
XAI_KEY = os.getenv("XAI_API_KEY")
ENDPOINT = "https://api.x.ai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {XAI_KEY}", "Content-Type": "application/json"}

def _grok_check(info: str, ground_truth: str) -> Optional[int]:
    if not XAI_KEY:
        return None
    prompt = (
        "You are a truth‑verification assistant.\n"
        "Determine whether the information given (user content + context) is sufficient to generate the ground‑truth answer.\n"
        "Return JSON {\"if_info_sufficient\": 1|0}.\n\n"
        f"Information:\n{info}\n\nGround truth:\n{ground_truth}"
    )
    msgs = [
        {"role": "system", "content": "You are a strict evaluator of information sufficiency."},
        {"role": "user", "content": prompt},
    ]
    try:
        r = requests.post(ENDPOINT, headers=HEADERS,
                          json={"model": "grok-3", "messages": msgs}, timeout=30)
        r.raise_for_status()
        resp = json.loads(r.json()["choices"][0]["message"]["content"])
        return int(resp.get("if_info_sufficient"))
    except Exception as e:
        LOG.error("Grok check failed: %s", e)
        return None

###############################################################################
# CSV iterator
###############################################################################

def _iter_csv(path: Path) -> Generator[Dict[str, Any], None, None]:
    with path.open("r", encoding="utf-8") as f:
        yield from csv.DictReader(f)

###############################################################################
# JSON helpers
###############################################################################

def _safe_json_parse(raw: str) -> Any:
    if not raw:
        raise ValueError("Empty field")
    if raw.startswith("\"") and raw.endswith("\""):
        try:
            raw = json.loads(raw)
        except Exception:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)

###############################################################################
# Normalize messages structure
###############################################################################

def _norm_content(items: List[Any]) -> List[Dict[str, Any]]:
    return [itm if isinstance(itm, dict) else {"type": "text", "text": str(itm)} for itm in items]


def _normalize_messages(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        out = []
        for itm in obj:
            if isinstance(itm, dict):
                c = itm.get("content")
                itm["content"] = _norm_content(c if isinstance(c, list) else [c])
                out.append(itm)
            else:
                out.append({"role": "user", "content": _norm_content([itm])})
        return out
    if isinstance(obj, dict):
        c = obj.get("content")
        obj["content"] = _norm_content(c if isinstance(c, list) else [c])
        return [obj]
    if isinstance(obj, str):
        return [{"role": "user", "content": _norm_content([obj])}]
    raise TypeError(type(obj))

###############################################################################
# Instruction + image extraction
###############################################################################

def _extract_instr_img(rec: Dict[str, Any]) -> tuple[str, str]:
    msgs_raw = rec.get("messages", "")
    try:
        msgs = _normalize_messages(_safe_json_parse(msgs_raw))
    except Exception:
        msgs = []
    instr = ""
    img_id = ""
    for m in msgs:
        if m.get("role") != "user":
            continue
        for p in m.get("content", []):
            if isinstance(p, dict):
                if p.get("type") == "text" and not instr:
                    instr = str(p.get("text", "")).strip()
                if p.get("type") == "image" and not img_id:
                    img_id = p.get("image")
        if instr:
            break
    if not instr:
        instr = str(rec.get("query", "")).strip()
    if not instr:
        raise ValueError("No instruction text located.")
    if not img_id:
        img_id = rec.get("image") or rec.get("session_id")
    return instr, img_id

###############################################################################
# Answer generators
###############################################################################

def _answer_simple(rec: Dict[str, Any]) -> str:
    return rec.get("ground_truth", "") if str(rec.get("is_correct")).lower() in {"true", "1", "yes"} else "I don't know."


def _answer_context(rec: Dict[str, Any], do_check: bool) -> str:
    # already correct
    if str(rec.get("is_correct")).lower() in {"true", "1", "yes"}:
        return rec.get("ground_truth", "")
    suff = str(rec.get("if_-1_is_info_enough_for_truth"))
    if suff not in {"0", "1"} and do_check:
        try:
            user_msg = next((m for m in _normalize_messages(_safe_json_parse(rec.get("messages", ""))) if m.get("role") == "user"), {})
            user_info = json.dumps(user_msg.get("content", ""), ensure_ascii=False)
        except Exception:
            user_info = rec.get("query", "")
        suff_val = _grok_check(user_info, rec.get("ground_truth", ""))
        if suff_val is not None:
            rec["if_-1_is_info_enough_for_truth"] = str(suff_val)
            suff = str(suff_val)
    if suff == "1":
        return f"Based on the provided context, {rec.get('ground_truth', '')}"
    return "I don't know."

###############################################################################
# Build chat object
###############################################################################

def _to_chat(rec: Dict[str, Any], mode: str, do_check: bool) -> List[Dict[str, Any]]:
    instr, img_id = _extract_instr_img(rec)
    ans = _answer_simple(rec) if mode == "simple" else _answer_context(rec, do_check)
    return [
        {"role": "user", "content": [{"type": "text", "text": instr}, {"type": "image", "image": img_id}]},
        {"role": "assistant", "content": [{"type": "text", "text": ans}]},
    ]

###############################################################################
# Script entry – breakpoint‑friendly
###############################################################################
DEFAULT_IN = "./turn_evaluation_results_all.csv"
DEFAULT_OUT = "./sft_data.jsonl"

parser = argparse.ArgumentParser(description="CRAG‑MM CSV → SFT JSONL")
parser.add_argument("--input", default=DEFAULT_IN)
parser.add_argument("--output", default=DEFAULT_OUT)
parser.add_argument("--answer_mode", choices=["simple", "context"], default="context")
parser.add_argument("--check_sufficiency", action="store_true", help="Call Grok when flag missing (context mode)")
args = parser.parse_args()

in_path = Path(args.input).expanduser()
out_path = Path(args.output).expanduser()
mode = args.answer_mode

do_check = bool(args.check_sufficiency) and mode == "context"
LOG.info("🔍 Reading   %s", in_path)
LOG.info("💾 Writing → %s", out_path)
LOG.info("📐 Answer mode      : %s", mode)
LOG.info("🔎 Sufficiency check : %s", "ENABLED" if do_check else "DISABLED")

total = ok = skipped = 0
with out_path.open("w", encoding="utf-8") as fout:
    for rec in _iter_csv(in_path):
        total += 1
        try:
            chat = _to_chat(rec, mode, do_check)
            fout.write(json.dumps(chat, ensure_ascii=False) + "\n")
            ok += 1
        except Exception as exc:
            LOG.error("🚫  Skipping row %d: %s", total, exc)
            skipped += 1

LOG.info("✅ Finished. kept=%d | skipped=%d | total=%d", ok, skipped, total)
