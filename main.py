#### All imports ####
from html import unescape
import os
import json
import asyncio
import logging
import re
from typing import List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketState

from utils.doc_processing_utils import html_to_text

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from doc_processing import DocumentProcessing
from utils.utils import create_return_response, json_default, to_data_url_from_b64, to_data_url_from_path
from fastapi.staticfiles import StaticFiles

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

app = FastAPI()

## preload on initialization ##
DP = DocumentProcessing(collection="docs_multimodal")
try:
    #warmup to reduce latency
    _ = DP._embed_query("warmup")
except Exception:
    pass

try:
    from openai import OpenAI
    OPENAI = OpenAI()
except Exception:
    OPENAI = None

## preload code end ##

FIG_DIR = "/Users/chinmay.deshmukh/Desktop/Projects/temp/EfficientRAG/figures"
app.mount("/figures", StaticFiles(directory=FIG_DIR), name="figures")

def to_url(image_path: str) -> str:
    basename = os.path.basename(image_path)
    return f"/figures/{basename}"


# CORS allow for streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# some initializations
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

#### Connection Manager ####
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info(f"New connection accepted: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logging.info(f"Connection closed: {websocket.client}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        # don’t send if the socket is closing/closed
        if websocket.application_state != WebSocketState.CONNECTED:
            return
        try:
            await websocket.send_text(message)
        except RuntimeError:
            pass
        except Exception as e:
            logging.warning(f"WS send failed: {e}")


manager = ConnectionManager()

#### Simple RAG hook ####
def _html_table_to_md_preview(html: str, max_rows: int = 8, max_cols: int = 6) -> str:
    """Tiny HTML Markdown preview"""
    try:
        import pandas as pd
        dfs = pd.read_html(html)
        if not dfs:
            return ""
        df = dfs[0].iloc[:max_rows, :max_cols]
        return df.to_markdown(index=False)
    except Exception:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.I | re.S)
        cells = [re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, flags=re.I | re.S) for r in rows]
        cleaned = [[re.sub(r"<.*?>", "", unescape(c)).strip() for c in row] for row in cells]
        cleaned = [r[:max_cols] for r in cleaned if any(x for x in r)]
        if not cleaned:
            return ""
        head = cleaned[0]
        body = cleaned[1:]
        out = ["| " + " | ".join(head) + " |",
               "| " + " | ".join(["---"] * len(head)) + " |"]
        for r in body[:max_rows-1]:
            out.append("| " + " | ".join(r) + " |")
        return "\n".join(out)

def _table_title_from_html(html: str) -> str:
    """Heuristic title from first header row/caption text."""
    try:
        import pandas as pd
        dfs = pd.read_html(html)
        if dfs:
            cols = [str(c) for c in dfs[0].columns.tolist()]
            return ", ".join(cols[:6])
    except Exception:
        pass
    # very light fallback
    m = re.search(r"<caption[^>]*>(.*?)</caption>", html, flags=re.I | re.S)
    if m:
        return re.sub(r"<.*?>", "", unescape(m.group(1))).strip()
    return "Untitled table"



async def retrieve_context(question: str, top_k: int = 20) -> Dict[str, Any]:
    results = DP.query(question, k=top_k, doc_id="doc-001")

    texts: List[str] = []
    images: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    for r in results:
        # figures -> images 
        if r.get("element_type") == "figure" and r.get("image_path"):
            data_url = to_data_url_from_b64(r["image_thumb_b64"]) if r.get("image_thumb_b64") else to_data_url_from_path(r["image_path"])
            images.append({
                "path": str(r["image_path"]),
                "caption": r.get("caption") or f"Figure p.{r.get('page','?')}",
                "data_url": data_url
            })

        # tables -> include html + preview + a human title
        if r.get("element_type") == "table" and r.get("table_html"):
            html = r["table_html"]
            preview = _html_table_to_md_preview(html)
            title = r.get("caption") or _table_title_from_html(html)
            tables.append({
                "page": r.get("page"),
                "title": title,
                "caption": r.get("caption"),
                "html": html,
                "preview_md": preview,
            })

        # context text for LLM
        for key in ("context", "text", "caption"):
            if r.get(key):
                texts.append(str(r[key]))
                break

    return {"texts": texts, "images": images, "tables": tables, "raw": results}



def build_messages_with_image_and_tables(question: str, contexts: List[str],
                                         image_data_url: str | None,
                                         table_previews_md: List[str]) -> List[Dict[str, Any]]:
    system_msg = (
        "You are a helpful assistant. Use the provided image(s), table previews, and text context to answer concisely. "
        "If the answer is not in the provided materials, say so briefly."
    )
    content_parts = [{"type": "text", "text": f"Question: {question}"}]

    if contexts:
        content_parts.append({"type": "text", "text": "[CONTEXT]\n" + "\n---\n".join(contexts[:4])})

    # include first image if present
    if image_data_url:
        content_parts.append({"type": "image_url", "image_url": {"url": image_data_url}})

    # include up to 2 table previews
    for i, md in enumerate(table_previews_md[:2], start=1):
        if md:
            content_parts.append({"type": "text", "text": f"[TABLE {i} PREVIEW]\n{md}"})

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": content_parts},
    ]

    

async def openai_answer(question: str, use_rag: bool) -> Dict[str, Any]:
    texts, images, tables = [], [], []
    image_data_url = None
    table_texts: List[str] = []

    if use_rag:
        rag = await retrieve_context(question)
        texts = rag.get("texts", [])
        images = [
            {"path": i.get("path"), "caption": i.get("caption")}
            for i in rag.get("images", [])
            if isinstance(i.get("path"), str) and i.get("path").strip()
        ]
        tables = [
            {"page": t.get("page"), "title": t.get("title"), "caption": t.get("caption"),
             "html": t.get("html"), "preview_md": t.get("preview_md")}
            for t in rag.get("tables", [])
        ]
        for img in rag.get("images", []):
            if img.get("data_url"):
                image_data_url = img["data_url"]; break
        table_texts = [t.get("preview_md") or html_to_text(t.get("html")) for t in tables]

    #hardcoding to search for table matches answer directly from tables we found
    q_low = question.lower()
    if any(k in q_low for k in ("what tables", "which tables", "list tables", "show tables", "tables do you have")) and tables:
        listing = "\n".join(
            [f"- p.{t['page']}: {t.get('title') or 'Untitled table'}" for t in tables]
        )
        answer_text = f"I found these tables in the document:\n{listing}"
        return {"answer": answer_text, "images": images, "tables": tables}

    messages = build_messages_with_image_and_tables(
        question=question,
        contexts=texts,
        image_data_url=image_data_url,
        table_previews_md=table_texts,
    )

    client = OpenAI()
    def _call():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
    resp = await asyncio.to_thread(_call)
    answer_text = resp.choices[0].message.content.strip()

    return {"answer": answer_text, "images": images, "tables": tables}



#### APIs ####
@app.post("/test", description="Test API to check if the server is running")
async def test():
    """A simple test endpoint."""
    return create_return_response(status_code=200, content="success")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)

                if data.get("type") == "hello":
                    await manager.send_personal_message(json.dumps({"system": "ready"}), websocket)
                    continue

                question: str = (data.get("question") or "").strip()
                use_rag: bool  = bool(data.get("use_rag", False))
                if not question:
                    await manager.send_personal_message(json.dumps({"error": "Please send a 'question' field."}), websocket)
                    continue

                answer = await openai_answer(question, use_rag)
                safe_payload = json.dumps(answer, default=json_default, ensure_ascii=False)
                await manager.send_personal_message(safe_payload, websocket)

            except json.JSONDecodeError:
                await manager.send_personal_message(json.dumps({"error": "Invalid JSON payload."}), websocket)
            except Exception as e:
                logging.exception("Error handling message")
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)