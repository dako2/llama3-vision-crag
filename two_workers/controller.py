import subprocess, json, sys, os, pathlib, tempfile

vision_p  = subprocess.Popen(
    ["python", "vision_worker.py"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
)
reason_p  = subprocess.Popen(
    ["python", "reason_worker.py"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
)

img = sys.argv[1]
question = " ".join(sys.argv[2:]) or "What is shown?"

# ---- caption ----
vision_p.stdin.write(json.dumps({"img": img}) + "\n"); vision_p.stdin.flush()
caption = json.loads(vision_p.stdout.readline())["caption"]
print("Caption:", caption)

# ---- reasoning ----
reason_p.stdin.write(json.dumps({"q": question, "cap": caption}) + "\n"); reason_p.stdin.flush()
answer = json.loads(reason_p.stdout.readline())["answer"]
print("Answer :", answer)
