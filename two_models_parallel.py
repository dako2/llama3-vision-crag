#!/usr/bin/env python3
# two_gpu_rag.py
import argparse, os, sys
from multiprocessing import Process, Pipe
from dataclasses import dataclass
from typing import Any
from PIL import Image
import json

# ─────────────────── configuration dataclasses ────────────────────
@dataclass
class LargeVisionCfg:
    model   : str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    gpu_util: float = 0.96
    tp      : int   = 1
    max_len : int   = 8192
    max_seqs: int   = 1

@dataclass
class ReasonCfg:
    model   : str = "meta-llama/Meta-Llama-3-8B-Instruct"
    gpu_util: float = 0.96
    tp      : int   = 1
    max_len : int   = 2048
    max_seqs: int   = 1

# ──────────────────── helper to build vLLM engine ──────────────────
def build_engine(cfg: Any):
    import vllm  # imported *after* CUDA_VISIBLE_DEVICES is set
    return vllm.LLM(
        model=cfg.model,
        tensor_parallel_size=cfg.tp,
        gpu_memory_utilization=cfg.gpu_util,
        max_model_len=cfg.max_len,
        max_num_seqs=cfg.max_seqs,
        trust_remote_code=True,
        dtype="bfloat16",
    )

# ────────────────── vision worker (caption generation) ─────────────
def vision_worker(child_conn, gpu_id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import vllm
    from PIL import Image

    engine = build_engine(LargeVisionCfg())
    tokenizer = engine.get_tokenizer()

    def caption(img_path: str) -> str:
        img = Image.open(img_path).convert("RGB").resize((960, 1280))
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Describe the image."},
                {"role": "user"  , "content": [{"type": "image"},
                                               {"type": "text", "text": ""}]},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        out = engine.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": img}}],
            sampling_params=vllm.SamplingParams(max_tokens=40, temperature=0.1)
        )[0].outputs[0].text.strip()
        return out

    # event loop
    while True:
        msg = child_conn.recv()
        if msg is None:  # shutdown signal
            break
        cap = caption(msg["img"])
        child_conn.send({"caption": cap})

# ───────────────── reasoning worker (Q+A) ──────────────────────────
def reason_worker(child_conn, gpu_id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import vllm

    engine = build_engine(ReasonCfg())
    tokenizer = engine.get_tokenizer()

    def answer(question: str, caption: str) -> str:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user"  , "content": f"Image caption: {caption}\n\n{question}"},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        out = engine.generate(
            [{"prompt": prompt}],
            sampling_params=vllm.SamplingParams(max_tokens=256, temperature=0.7)
        )[0].outputs[0].text.strip()
        return out

    # event loop
    while True:
        msg = child_conn.recv()
        if msg is None:
            break
        ans = answer(msg["q"], msg["cap"])
        child_conn.send({"answer": ans})

# ────────────────────────── main script ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to image file")
    ap.add_argument("--q",   required=True, help="Question about the image")
    ap.add_argument("--vision-gpu", type=int, default=1, help="GPU id for vision model")
    ap.add_argument("--reason-gpu", type=int, default=0, help="GPU id for reasoning model")
    args = ap.parse_args()

    # set start method early
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # vision subprocess
    v_parent, v_child = Pipe()
    vision_proc = Process(target=vision_worker, args=(v_child, args.vision_gpu))
    vision_proc.start()

    # send image → get caption
    v_parent.send({"img": args.img})
    caption = v_parent.recv()["caption"]
    print(f"\n📸 Caption: {caption}", flush=True)

    # reasoning subprocess
    r_parent, r_child = Pipe()
    reason_proc = Process(target=reason_worker, args=(r_child, args.reason_gpu))
    reason_proc.start()

    # send caption + question → get answer
    r_parent.send({"q": args.q, "cap": caption})
    answer = r_parent.recv()["answer"]
    print(f"\n🧠 Answer : {answer}")

    # graceful shutdown
    v_parent.send(None); r_parent.send(None)
    vision_proc.join();   reason_proc.join()

if __name__ == "__main__":
    main()
