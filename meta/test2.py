from huggingface_hub import snapshot_download
import spacy

# 1. Pull the entire en_core_web_sm repo locally
model_dir = snapshot_download(repo_id="spacy/en_core_web_sm")

# 2. Load from that directory just like any on-disk spaCy model
nlp = spacy.load(model_dir)
