# vision_worker.py
import os, sys, json, vllm
from PIL import Image
from dataclasses import dataclass

os.environ["CUDA_VISIBLE_DEVICES"] = "1"          # <-- lock to GPU-1 before torch import

@dataclass
class LargeVisionCfg:
    model      = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    gpu_util   = 0.96
    tp         = 1
    max_len    = 1536
    max_seqs   = 1

engine = vllm.LLM(
    model                 = LargeVisionCfg.model,
    tensor_parallel_size  = LargeVisionCfg.tp,
    gpu_memory_utilization= LargeVisionCfg.gpu_util,
    max_model_len         = LargeVisionCfg.max_len,
    max_num_seqs          = LargeVisionCfg.max_seqs,
    trust_remote_code     = True,
    dtype                 = "bfloat16",
)
tok = engine.get_tokenizer()

def caption(img_path: str) -> str:
    img = Image.open(img_path).convert("RGB").resize((960, 1280))
    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": "Describe the image."},
            {"role": "user"  , "content": [{"type": "image"},
                                           {"type": "text", "text": ""}]}
        ],
        add_generation_prompt=True,
        tokenize=False
    )
    out = engine.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": img}}],
        sampling_params=vllm.SamplingParams(max_tokens=40, temperature=0.1)
    )[0].outputs[0].text.strip()
    return out

# ------- simple stdin/stdout protocol ----------
for line in sys.stdin:                       # expect {"img": "..."}
    req = json.loads(line)
    cap = caption(req["img"])
    print(json.dumps({"caption": cap}), flush=True)
