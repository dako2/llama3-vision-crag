# remote_search_pipeline.py
import requests, base64, io
from typing import Any, List
from PIL import Image

class RemoteSearchPipeline:
    """
    Makes HTTP calls to the search service but
    behaves like UnifiedSearchPipeline locally.
    """
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.text_ep  = f"{base_url}/search/text"
        self.image_ep = f"{base_url}/search/image"

    # keep the *exact* call style:  obj(input, k=5)
    def __call__(self, query_or_image: Any, k: int = 5) -> List[dict]:
        # ---------- text -------------
        if isinstance(query_or_image, str):
            payload = {"query": query_or_image, "top_k": k}
            r = requests.post(self.text_ep, json=payload, timeout=15)
        # ---------- image ------------
        else:
            # accept PIL.Image, bytes, or ndarray
            if isinstance(query_or_image, Image.Image):
                buf = io.BytesIO()
                query_or_image.save(buf, format="PNG")
                img_bytes = buf.getvalue()
            elif isinstance(query_or_image, (bytes, bytearray)):
                img_bytes = bytes(query_or_image)
            else:  # e.g. numpy array
                raise TypeError("Unsupported image type; pass PIL.Image or bytes.")
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            payload = {"image_base64": img_b64, "top_k": k}
            r = requests.post(self.image_ep, json=payload, timeout=30)

        r.raise_for_status()
        return r.json()["results"]
