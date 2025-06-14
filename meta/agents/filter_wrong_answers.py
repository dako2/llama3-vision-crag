import json
import os
import logging
import requests
from dotenv import load_dotenv

INPUT_FILE = "selected_pipeline_finetune_data.jsonl"
OUTPUT_FILE = "selected_pipeline_finetune_data_wrong_asnwers_with_flag.jsonl"
LOG_FILE = "grok_info_sufficiency.log"

# === Logging Setup ===
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("🔍 Starting enrichment process...")

# === Load Grok API Key ===
load_dotenv()
XAI_KEY = os.getenv("XAI_API_KEY")
ENDPOINT = "https://api.x.ai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {XAI_KEY}",
    "Content-Type": "application/json"
}

# === Grok Request Function ===
def grok_check_information_sufficiency(info: str, ground_truth: str) -> str:
    prompt = (
        "You are a truth verification assistant.\n"
        "Determine whether the information given (user content and context) is sufficient to generate the ground truth answer.\n"
        "Return a JSON with a single key 'if_info_sufficient', value must be:\n"
        "- 1: if the given information is enough to generate the ground truth\n"
        "- 0: if it's not enough\n\n"
        f"Information:\n{info}\n\n"
        f"Ground truth:\n{ground_truth}"
    )
    messages = [
        {"role": "system", "content": "You are a helpful and strict evaluator for determining information sufficiency."},
        {"role": "user", "content": prompt}
    ]
    response = requests.post(ENDPOINT, headers=HEADERS, json={"model": "grok-3", "messages": messages}, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# === Main Processing Function ===
def enrich_and_append(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "a", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            accuracy = data.get("accuracy", "n/a")

            if accuracy != -1:
                data["if_-1_is_info_enough_for_truth"] = "n/a"
            else:
                try:
                    assistant_msg = next((m for m in data.get("messages", []) if m.get("role") == "assistant"), {})
                    user_msg = next((m for m in data.get("messages", []) if m.get("role") == "user"), {})
                    assistant_answer = assistant_msg.get("content", "")
                    user_info = user_msg.get("content", "")
                    ground_truth = data.get("ground_truth", "")

                    response = grok_check_information_sufficiency(user_info, ground_truth)
                    logger.info(f"🧾 Ground Truth: {ground_truth}")
                    logger.info(f"📥 Grok Response: {response}")

                    parsed = json.loads(response)
                    suff = parsed.get("if_info_sufficient")
                    data["if_-1_is_info_enough_for_truth"] = suff if suff in [0, 1] else "n/a"
                except Exception as e:
                    logger.error(f"❌ Exception for session_id={data.get('session_id', 'unknown')}: {e}")
                    data["if_-1_is_info_enough_for_truth"] = "n/a"

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    enrich_and_append(INPUT_FILE, OUTPUT_FILE)
    logger.info(f"✅ Completed and appended enriched lines to {OUTPUT_FILE}")
