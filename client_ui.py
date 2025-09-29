import json
from typing import Any, Dict, List
import websocket
import streamlit as st

## global css start ##
st.set_page_config(page_title="RAG Chat (WebSocket)", page_icon="💬", layout="centered")
st.title("RAG Chat (WebSocket)")
st.markdown("""
<style>
.stImage img { max-width: 640px; height: auto; }  /* keep images reasonable */
</style>
""", unsafe_allow_html=True)
## global css end ##

## controls start ##
server_url = st.text_input("WebSocket Server URL", value="ws://localhost:9999/ws", key="ws_url")
use_rag    = st.checkbox("Use RAG (documents)", value=True, key="use_rag")
## cpmtrols end ##

## helper functions start ##
def render_table_block(table: Dict[str, Any], key_prefix: str) -> None:
    """Render a table from backend (prefer DataFrame; fallback to markdown/HTML)."""
    label = table.get("caption") or table.get("title") or "Table"
    html = table.get("html")
    preview_md = (table.get("preview_md") or "").strip()

    if html:
        try:
            import pandas as pd
            dfs = pd.read_html(html)
            if dfs:
                st.caption(label)
                st.dataframe(dfs[0], use_container_width=True, key=f"{key_prefix}_df")
                return
        except Exception:
            pass

    if preview_md:
        st.caption(label)
        st.markdown(preview_md)
        return

    if html:
        st.caption(label)
        st.components.v1.html(html, height=260, scrolling=True)

def ask_once(ws_url: str, payload: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
    """Open a WS, perform a hello handshake, send payload, wait for real answer, close."""
    ws = websocket.create_connection(ws_url, timeout=timeout)
    ws.settimeout(timeout)

    # handshake (ignore pings until {"system":"ready"})
    ws.send(json.dumps({"type": "hello"}))
    while True:
        raw = ws.recv()
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if msg.get("system") == "ready":
            break
        if "ping" in msg:
            continue

    # send real question
    ws.send(json.dumps(payload))

    # read frames until the actual answer (ignore ping/system)
    data: Dict[str, Any] = {}
    while True:
        raw = ws.recv()
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if "ping" in msg or "system" in msg:
            continue
        data = msg
        break

    try:
        ws.close()
    except Exception:
        pass
    return data

## helper functions start ##

## chat state ##
if "messages" not in st.session_state:
    # each: {"role": "user"/"assistant", "text": str, "images": [...], "tables": [...]}
    st.session_state.messages: List[Dict[str, Any]] = []

## history renderer start ##
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg.get("role", "assistant")):
        if msg.get("text"):
            st.markdown(msg["text"])
        for j, img in enumerate(msg.get("images", [])):
            path = (img.get("path") or "").strip()
            if path:
                st.image(path, caption=img.get("caption") or None, use_container_width=True)
        for k, tbl in enumerate(msg.get("tables", [])):
            render_table_block(tbl, key_prefix=f"tbl_{i}_{k}")

## history renderer end ##

## start the ws when we receive the prompt and process the response from DP ##
prompt = st.chat_input("Type your question…")
if prompt:
    # show user bubble immediately
    st.session_state.messages.append({"role": "user", "text": prompt, "images": [], "tables": []})

    with st.spinner("Thinking…"):
        try:
            resp = ask_once(server_url, {"question": prompt, "use_rag": use_rag})
        except Exception as e:
            resp = {"answer": f"WS error: {e}", "images": [], "tables": []}

    # parse response
    text = resp.get("answer", "") or ""
    imgs = [
        {"path": i.get("path"), "caption": i.get("caption")}
        for i in (resp.get("images", []) or [])
        if isinstance(i, dict) and isinstance(i.get("path"), str) and i.get("path").strip()
    ]
    tables = [
        {
            "page": t.get("page"),
            "title": t.get("title"),
            "caption": t.get("caption"),
            "html": t.get("html"),
            "preview_md": t.get("preview_md"),
        }
        for t in (resp.get("tables", []) or [])
        if isinstance(t, dict)
    ]

    st.session_state.messages.append({"role": "assistant", "text": text, "images": imgs, "tables": tables})
    st.rerun()