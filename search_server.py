# search_server.py
import io, base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from cragmm_search.search import UnifiedSearchPipeline

pipeline = UnifiedSearchPipeline(
    text_model_name="BAAI/bge-large-en-v1.5",
    image_model_name="openai/clip-vit-large-patch14-336",
    web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
)

app = FastAPI(title="CRAG-MM Search API")

# ---------- request schemas ----------
class TextReq(BaseModel):
    query: str
    top_k: int = 5

class ImageReq(BaseModel):
    image_base64: str        # PNG/JPG bytes, base64-encoded
    top_k: int = 5
# -------------------------------------

@app.post("/search/text")
def text_search(req: TextReq):
    hits = pipeline(req.query, k=req.top_k)
    return {"results": hits}

@app.post("/search/image")
def image_search(req: ImageReq):
    try:
        img_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image data: {e}")
    hits = pipeline(image, k=req.top_k)
    return {"results": hits}

#uvicorn search_server:app --host 0.0.0.0 --port 8001
