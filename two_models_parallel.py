import argparse, torch, vllm
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class VConfig:
    model: str
    gpu_util: float = 0.92
    tp: int = 1
    max_len: int = 4096
    max_seqs: int = 256

def build_engine(cfg: VConfig) -> vllm.LLM:
    return vllm.LLM(
        model=cfg.model,
        tensor_parallel_size=cfg.tp,
        gpu_memory_utilization=cfg.gpu_util,
        max_model_len=cfg.max_len,
        max_num_seqs=cfg.max_seqs,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
    )

def caption(img: Image.Image, engine: vllm.LLM) -> str:
    tok = engine.get_tokenizer()
    messages = [
        {"role": "system", "content": "Describe the image."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ""}]},
    ]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    res = engine.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": img}}],
        sampling_params=vllm.SamplingParams(max_tokens=40, temperature=0.1)
    )[0].outputs[0].text.strip()
    return res

def reason(q: str, cap: str, engine: vllm.LLM) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Image caption: {cap}\n\n{q}"}
    ]
    tok = engine.get_tokenizer()
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    res = engine.generate(
        prompt,
        sampling_params=vllm.SamplingParams(max_tokens=256, temperature=0.7)
    )[0].outputs[0].text.strip()
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--q",   required=True)
    args = ap.parse_args()

    # 1️⃣  Caption with vision model then unload
    vision_engine = build_engine(VConfig("./llama3-vision-11b", gpu_util=0.6))
    cap = caption(Image.open(args.img), vision_engine)
    print("🔎 Caption:", cap)
    del vision_engine; torch.cuda.empty_cache()

    # 2️⃣  Reason with text-only model
    text_engine = build_engine(VConfig("./llama3-text-8b", gpu_util=0.9))
    ans = reason(args.q, cap, text_engine)
    print("🧠 Answer :", ans)
