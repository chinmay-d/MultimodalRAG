from pydantic import BaseModel
from typing import Optional, Tuple

class ParsedElement(BaseModel):
    doc_id: str
    page: int
    element_type: str # paragraph/table/figure
    text: str # normalized text
    table_html: Optional[str] = None # if table
    caption: Optional[str] = None # if figure
    bbox: Optional[Tuple[float,float,float,float]] = None
    order: Optional[int] = None
    image_bytes: Optional[bytes] = None # for figures/pages
    image_path: Optional[str] = None # path of the image