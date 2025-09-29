from typing import Iterable
from unstructured.documents.elements import Table as UTable, Element


def batch_iter(it: Iterable, batch_size: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf

def html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ").strip()
    except Exception:
        return html
    
def linearize_table(ut: UTable) -> str:
        txt = (ut.text or "").strip()
        if not txt:
            # fallback to HTML text if available
            html = getattr(ut, "text_as_html", None) or getattr(ut.metadata, "text_as_html", None) or ""
            return html_to_text(html)
        return txt
