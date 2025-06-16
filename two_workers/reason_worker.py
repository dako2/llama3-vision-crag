# reason_worker.py
import os, sys, json, vllm
from dataclasses import dataclass

os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # <-- lock to GPU-0

@dataclass
class ReasonCfg:
    model      = "meta-llama/Meta-Llama-3-8B-Instruct"
    gpu_util   = 0.96
    tp         = 1
    max_len    = 2048
    max_seqs   = 75

engine = vllm.LLM(
    model                 = ReasonCfg.model,
    tensor_parallel_size  = ReasonCfg.tp,
    gpu_memory_utilization= ReasonCfg.gpu_util,
    max_model_len         = ReasonCfg.max_len,
    max_num_seqs          = ReasonCfg.max_seqs,
    trust_remote_code     = True,
    dtype                 = "bfloat16",
)
tok = engine.get_tokenizer()

def answer(question: str, caption: str) -> str:
    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user"  , "content": f"Image caption: {caption}\n\n{question}"},
        ],
        add_generation_prompt=True,
        tokenize=False
    )
    out = engine.generate(
        [{"prompt": prompt}],
        sampling_params=vllm.SamplingParams(max_tokens=256, temperature=0.7)
    )[0].outputs[0].text.strip()
    return out

# ------- simple stdin/stdout protocol ----------
for line in sys.stdin:                       # expect {"q": "...", "cap": "..."}
    req = json.loads(line)
    ans = answer(req["q"], req["cap"])
    print(json.dumps({"answer": ans}), flush=True)
