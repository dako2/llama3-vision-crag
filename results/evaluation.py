import os, requests, json
from typing import Any, Dict
from datasets import Dataset
from dotenv import load_dotenv

# === Config ===
JSONL_FILE = "./your_predictions.jsonl"  # ✅ path to your prediction .jsonl

# === Grok API Config ===
load_dotenv()
XAI_KEY = os.getenv("XAI_API_KEY")
ENDPOINT = "https://api.x.ai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {XAI_KEY}",
    "Content-Type": "application/json"
}

# === Evaluation Prompt ===
EVAL_SYSTEM_PROMPT = (
    "You are an expert evaluator for a visual question-answering system. "
    "Your task is to classify the prediction's correctness based on the ground truth and user query.\n\n"
    "Output a JSON object with a single key 'accuracy', whose value MUST be one of:\n"
    "- 1: If the prediction is **correct** — it captures all key information from the ground truth, even if phrased differently.\n"
    "- -1: If the prediction is **incorrect** — it misses essential details or contains wrong information.\n"
    "- 0: If the prediction clearly says '**I don’t know**' or equivalent phrasing (e.g. 'No idea.').\n"
    "- -0.5: If the prediction expresses **uncertainty** (e.g., 'I'm not sure', 'It depends', 'Possibly') without giving incorrect facts.\n\n"
    "Base your judgment strictly on the alignment with the ground truth. Do NOT infer from the image, only what's written."
)

# === Grok Chat Function ===
def grok_chat(messages, model="grok-3", **params):
    body = {"model": model, "messages": messages, **params}
    r = requests.post(ENDPOINT, headers=HEADERS, json=body, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# === Evaluator Class ===
class CRAGEvaluator:
    def __init__(self, dataset: Dataset = None, eval_model_name: str = "grok-3"):
        self.eval_model_name = eval_model_name
        self.dataset_lookup = {}

        if dataset is not None:
            for entry in dataset["answers"]:
                session_ids = entry.get("interaction_id", [])
                for sid in session_ids:
                    self.dataset_lookup[sid] = entry


    def evaluate_one_jsonl_line(self, jsonl_line: Dict[str, Any]) -> Dict[str, float]:
        session_id = jsonl_line.get("session_id")
        messages = jsonl_line.get("messages", [])

        # Extract assistant response
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        agent_response = assistant_msg["content"].strip() if assistant_msg else ""

        # Extract user query
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        query = ""
        if user_msg:
            parts = user_msg.get("content", [])
            query = next((p["text"] for p in parts if isinstance(p, dict) and p.get("type") == "text"), "")

        # ✅ Lookup ground truth via session_id (now using interaction_id)
        gt_entry = self.dataset_lookup.get(session_id)
        if not gt_entry:
            print(f"[Eval] Missing ground truth for session_id: {session_id}")
            return {"accuracy": -1}

        ground_truths = gt_entry.get("ans_full", [])
        ground_truth = ground_truths[0] if ground_truths else ""

        # Rule 0: Explicit IDK check
        if agent_response.lower().strip() in {"i don't know.", "i don't know", "no idea", "i have no idea"}:
            return {"accuracy": 0}

        # Rule 1: Exact match
        if agent_response.strip().lower() == ground_truth.strip().lower():
            return {"accuracy": 1}

        # Rule 2–4: Grok evaluation
        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {agent_response}\n"
            },
        ]

        try:
            response = grok_chat(messages, model=self.eval_model_name, temperature=0.0)
            parsed = json.loads(response)
            accuracy = parsed.get("accuracy")
            if accuracy in {1, -1, 0, -0.5}:
                return {"accuracy": accuracy}
            else:
                print(f"[Eval] Invalid accuracy value returned: {accuracy}")
                return {"accuracy": -1}
        except Exception as exc:
            print(f"[Eval] Grok error for session {session_id}: {exc}")
            return {"accuracy": -1}
    
    @classmethod
    def from_jsonl(cls, gt_jsonl_path: str, eval_model_name: str = "grok-3"):
        """
        Create a CRAGEvaluator from a local .jsonl ground truth file
        that contains {"session_id": ..., "ans_full": [...]} per line.
        """
        dataset_lookup = {}
        with open(gt_jsonl_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                entry = json.loads(line)
                sid = entry.get("session_id")
                if sid:
                    dataset_lookup[sid] = entry

        # Create an instance and override lookup dictionary
        instance = cls(dataset=None, eval_model_name=eval_model_name)
        instance.dataset_lookup = dataset_lookup
        return instance