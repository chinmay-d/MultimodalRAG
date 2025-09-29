

#### utility functions ####
import base64
import mimetypes
import numpy as np


def create_return_response(status_code, content):
    return {"status_code": status_code, "content": content}

def json_default(o):
    # Convert numpy scalars/arrays to plain Python for json.dumps
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    # Fallback
    try:
        return str(o)
    except Exception:
        return None
    

def to_data_url_from_b64(b64_str: str, mime: str = "image/jpeg") -> str:
    b64_clean = b64_str.strip()
    return f"data:{mime};base64,{b64_clean}"

def to_data_url_from_path(path: str) -> str | None:
    try:
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            b = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b}"
    except Exception:
        return None
    
