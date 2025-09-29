from __future__ import annotations
from collections import defaultdict
import os
import io
import uuid
import json
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Torch / Transformers
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoProcessor


# Sentence-Transformers
from sentence_transformers import SentenceTransformer, CrossEncoder

# Unstructured parsing
from unstructured.partition.auto import partition
from unstructured.documents.elements import Table as UTable, Element

from schemas.document_processing_schemas import ParsedElement
from utils.doc_processing_utils import batch_iter, html_to_text, linearize_table
try:
    from unstructured.documents.elements import Image as UImage
except Exception:
    UImage = None


#### Models we'll be usign ####
TEXT_EMBED_MODEL = "BAAI/bge-m3" #for dense texst
RERANK_MODEL     = "BAAI/bge-reranker-v2-m3" # crossencoder reranker
SIGLIP_MODEL     = "google/siglip-base-patch16-224" # image embedding (dense)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.getenv("QDRANT_API_KEY", None)

## vectors
VECTOR_SPECS = {
    "text_dense": 1024, # bge-m3 output
    "image_dense": 768, # siglip-base-patch16-224 output dims
}

DISTANCE = {
    "text_dense": qmodels.Distance.COSINE,
    "image_dense": qmodels.Distance.COSINE,
}


class DocumentProcessing:
    def __init__(
        self, 
        collection: str = "docs_multimodal",
        device: Optional[str] = None,
    ):
        """
        Initializes 
        1. TEXT_EMBED_MODEL -> for creating text embeds
        2. RERANK -> for rerankng
        3. SIGLIP -> for image encodings

        It also creates collection if it doesn't exist
        """
        # ---- device selection (CUDA > MPS > CPU) ----
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda" # for nvidia
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps" # for mac
            else:
                self.device = "cpu" # for all other systems

        self.collection = collection # collection name

        # Where to store extracted figure images 
        self.assets_dir = os.path.abspath(os.getenv("RAG_ASSETS_DIR", "./data/images"))
        os.makedirs(self.assets_dir, exist_ok=True)

        # Qdrant client 
        self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, check_compatibility=False)

        #### Load Models ####
        ## 1. Text Embedding (dense) 
        self.text_model = SentenceTransformer(TEXT_EMBED_MODEL, device=self.device)

        ## 2. Image embedding SigLIP 
        self.siglip_processor = AutoProcessor.from_pretrained(
            SIGLIP_MODEL, use_fast=True, trust_remote_code=True
        )
        self.siglip_model = AutoModel.from_pretrained(
            SIGLIP_MODEL, trust_remote_code=True
        ).to(self.device)
        self.siglip_model.eval()

        ## 3. Reranker
        self.reranker = CrossEncoder(RERANK_MODEL, device=self.device)

        # Create qdrant collection if missing
        self._ensure_collection()


    def _ensure_collection(self):
        # Detect existing collections
        try:
            existing = self.qdrant.get_collections().collections
            names = {c.name for c in existing}
        except Exception:
            names = set()

        if self.collection in names:
            return

        # Build named vectors as a plain dict -> works across old/new clients
        vector_params = {
            name: qmodels.VectorParams(size=size, distance=DISTANCE[name])
            for name, size in VECTOR_SPECS.items()
        }

        # Prepare kwargs gradually; older servers reject unknown args, so we try minimal first
        kwargs = dict(
            collection_name=self.collection,
            vectors_config=vector_params, 
        )

        # Attempt creation with rich config if it fails, fall back to minimal args
        try:
            self.qdrant.recreate_collection(**kwargs)
        except Exception:
            # Minimal fallback just vectors_config
            self.qdrant.recreate_collection(
                collection_name=self.collection,
                vectors_config=vector_params,
            )

        # Besteffort payload indexes
        index_specs = [("doc_id", "KEYWORD"), ("page", "INTEGER"), ("element_type", "KEYWORD"), ("order", "INTEGER")]
        schema_type = getattr(qmodels, "PayloadSchemaType", None)
        for key, typ in index_specs:
            try:
                if schema_type and hasattr(schema_type, "__members__"):
                    schema = schema_type.__members__[typ]
                elif schema_type:
                    schema = schema_type(typ)
                else:
                    schema = None
                if schema:
                    self.qdrant.create_payload_index(self.collection, key, schema)
            except Exception:
                pass


    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        # bge-m3 expects: "passage: ..." for passages
        inputs = [f"passage: {t}" for t in texts]
        embs = self.text_model.encode(inputs, normalize_embeddings=True, convert_to_numpy=True)
        return embs.tolist()

    def _embed_query(self, q: str) -> List[float]:
        # bge-m3 expects: "query: ..." for queries
        return self.text_model.encode([f"query: {q}"], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
    
    @torch.no_grad()
    def _embed_query_for_image(self, q: str) -> List[float]:
        """
        Use SigLIP's text tower to embed the query so we can retrieve images (cross modal).
        """
        proc = self.siglip_processor
        if hasattr(proc, "tokenizer"):
            tokens = proc.tokenizer([q], return_tensors="pt")
        else:
            tokens = proc(text=[q], return_tensors="pt")

        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        text_feats = self.siglip_model.get_text_features(**tokens)
        text_feats = torch.nn.functional.normalize(text_feats, dim=-1)
        return text_feats[0].detach().cpu().tolist()
    
    def _save_image_bytes(self, raw: bytes, doc_id: str, page: int, order: int) -> Tuple[str, Optional[str]]:
        """
        Save full image to disk and also return a tiny base64 thumbnail for quick previews.
        Returns (image_path, image_thumb_b64).
        """
        try:
            from PIL import Image
            import base64
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            # filename
            fname = f"{doc_id}_p{page}_o{order}.jpg"
            fpath = os.path.join(self.assets_dir, fname)
            img.save(fpath, format="JPEG", quality=90)

            # small thumbnail (max side 256)
            im2 = img.copy()
            im2.thumbnail((256, 256))
            buf = io.BytesIO()
            im2.save(buf, format="JPEG", quality=75)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return fpath, b64
        except Exception:
            return "", None
        

    @torch.no_grad()
    def _embed_image(self, image_bytes: bytes) -> List[float]:
        if not image_bytes:
            return []
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.siglip_processor(images=img, return_tensors="pt").to(self.device)
        out = self.siglip_model.get_image_features(**inputs)
        out = torch.nn.functional.normalize(out, dim=-1)
        return out[0].detach().cpu().tolist()


    def _compute_vectors_for_element(self, el: ParsedElement):
        vectors = {}

        # text vectors (paragraphs, tables, figures)
        if el.element_type in ("paragraph", "table", "figure"):
            base_text = el.text or ""
            
            # add table HTML as extra context
            if el.element_type == "table" and el.table_html:
                from utils.doc_processing_utils import html_to_text
                base_text += "\n" + html_to_text(el.table_html)

            td = self._embed_text([base_text])[0]
            vectors["text_dense"] = td


            # image vectors
            if el.element_type == "figure":
                img_vec = []
                if el.image_bytes:
                    img_vec = self._embed_image(el.image_bytes)
                elif el.image_path and os.path.exists(el.image_path):
                    try:
                        with open(el.image_path, "rb") as f:
                            img_vec = self._embed_image(f.read())
                    except Exception:
                        img_vec = []

                if img_vec:
                    vectors["image_dense"] = img_vec

            return vectors
    

    def _save_image_bytes(self, raw: bytes, doc_id: str, page: int, order: int) -> Tuple[str, Optional[str]]:
        """
        Save full image to disk and also return a tiny base64 thumbnail for quick previews.
        Returns (image_path, image_thumb_b64).
        """
        try:
            from PIL import Image
            import base64
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            # filename
            fname = f"{doc_id}_p{page}_o{order}.jpg"
            fpath = os.path.join(self.assets_dir, fname)
            img.save(fpath, format="JPEG", quality=90)

            # small thumbnail size 256 x 256
            im2 = img.copy()
            im2.thumbnail((256, 256))
            buf = io.BytesIO()
            im2.save(buf, format="JPEG", quality=75)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return fpath, b64
        except Exception:
            return "", None
    

    def _element_payload(self, el: ParsedElement) -> Dict[str, Any]:
        payload = {
            "doc_id": el.doc_id,
            "page": el.page,
            "element_type": el.element_type,
            "text": el.text,
            "caption": el.caption,
            "table_html": el.table_html,
            "order": el.order,
        }
        if el.bbox:
            payload["bbox"] = el.bbox

        if el.element_type == "figure":
            image_path = el.image_path
            thumb_b64 = None

            # if unstructured already saved a file, reuse it
            if image_path and os.path.exists(image_path):
                # make a thumbnail
                try:
                    from PIL import Image
                    import base64
                    img = Image.open(image_path).convert("RGB")
                    im2 = img.copy()
                    im2.thumbnail((256, 256))
                    buf = io.BytesIO()
                    im2.save(buf, format="JPEG", quality=75)
                    thumb_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                except Exception:
                    thumb_b64 = None

            # else save our bytes copy into assets_dir
            elif el.image_bytes:
                image_path, thumb_b64 = self._save_image_bytes(
                    el.image_bytes, el.doc_id, el.page, el.order or 0
                )

            if image_path:
                payload["image_path"] = image_path
            if thumb_b64:
                payload["image_thumb_b64"] = thumb_b64

        return payload
    
    def _token_chunks(self, text: str, target_tokens: int = 300, overlap: int = 30) -> List[str]:
        """
        Approximate token based splitting using whitespace count (fast and good enough).
        Swap with tiktoken if you need exact counts.
        """
        words = text.split()
        if len(words) <= target_tokens:
            return [text]

        chunks = []
        step = target_tokens - overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + target_tokens]
            chunks.append(" ".join(chunk_words))
            if i + target_tokens >= len(words):
                break
        return chunks


    def _parse_and_chunk(self, file_path: str, doc_id: str) -> List[ParsedElement]:
        """
        Uses unstructured (hi_res) to detect layout elements.
        We chunk by layout: paragraphs (~200–350 tokens), tables (as one unit), figures (caption + image).
        """
        # Unstructured smart partition
        parts: List[Element] = partition(filename=file_path, strategy="hi_res", extract_images_in_pdf=True, image_output_dir_path=self.assets_dir, infer_table_structure=True)

        elements: List[ParsedElement] = []
        order = 0

        for p in parts:
            order += 1
            page = getattr(p.metadata, "page_number", None) or 1
            bbox = None
            if getattr(p, "coordinates", None) and p.coordinates and p.coordinates.points:
                xs = [pt.x for pt in p.coordinates.points]
                ys = [pt.y for pt in p.coordinates.points]
                bbox = (min(xs), min(ys), max(xs), max(ys))

            # Table
            if isinstance(p, UTable):
                table_html = getattr(p, "text_as_html", None) or getattr(p.metadata, "text_as_html", None)
                linear = linearize_table(p)

                combined_text = linear

                if elements and elements[-1].element_type == 'paragraph':
                    # Prepend the last paragraph's text as a contextual header
                    combined_text = elements[-1].text + "\n\n" + linear
                
                elements.append(ParsedElement(
                    doc_id=doc_id, page=page, element_type="table",
                    text=combined_text,  # Use the new, context-rich text
                    table_html=table_html or "", bbox=bbox, order=order
                ))
                continue

            # Image / figure detection (handles base64 or ondisk paths) 
            is_image_elem = (UImage is not None and isinstance(p, UImage))

            # Unstructured versions vary check multiple keys
            meta = getattr(p, "metadata", None)
            has_b64  = bool(getattr(meta, "image_base64", None)) if meta else False

            # Accept any of these path styles:
            cand_paths = []
            if meta:
                for key in ("image_path", "image_paths", "image_filename", "image_file_path"):
                    val = getattr(meta, key, None)
                    if isinstance(val, str) and val:
                        cand_paths.append(val)
                    elif isinstance(val, list):
                        cand_paths.extend([v for v in val if isinstance(v, str) and v])

            # Decide if this element is an image at all
            has_path = len(cand_paths) > 0
            if is_image_elem or has_b64 or has_path:
                raw = None
                img_path = None

                # Try to resolve a usable path if any were provided
                if cand_paths:
                    # try a few locations for each candidate
                    pdf_dir = os.path.dirname(os.path.abspath(file_path))
                    cwd = os.getcwd()
                    for rel in cand_paths:
                        candidates = []

                        # If already absolute, try as is
                        if os.path.isabs(rel):
                            candidates.append(rel)
                        else:
                            candidates.append(os.path.join(cwd, rel))
                            candidates.append(os.path.join(pdf_dir, rel))
                            base = os.path.basename(rel)
                            candidates.append(os.path.join(cwd, "figure", base))
                            candidates.append(os.path.join(cwd, "data/images", base))
                            candidates.append(os.path.join(self.assets_dir, base))

                        found = next((c for c in candidates if os.path.exists(c)), None)
                        if found:
                            img_path = os.path.abspath(found)
                            break

                # If no working path, try base64 bytes
                if img_path is None and has_b64:
                    try:
                        import base64
                        raw = base64.b64decode(meta.image_base64)
                    except Exception:
                        raw = None

                caption = (p.text or "").strip() or None
                elements.append(ParsedElement(
                    doc_id=doc_id,
                    page=page,
                    element_type="figure",
                    text=caption or "",
                    caption=caption,
                    bbox=bbox,
                    order=order,
                    image_bytes=raw,
                    image_path=img_path,
                ))
                continue

            # para like text (headers, narrative)
            text = (p.text or "").strip()
            if not text:
                continue

            # split long paragraphs into ~200–350 token chunks with small overlap
            for chunk in self._token_chunks(text, target_tokens=300, overlap=30):
                elements.append(ParsedElement(
                    doc_id=doc_id, page=page, element_type="paragraph",
                    text=chunk, bbox=bbox, order=order
                ))

        # ensure deterministic order for sibling context expansion
        elements.sort(key=lambda e: (e.page, e.order))
        return elements


    def ingest(self, file_path: str, doc_id: str, batch_size: int = 128) -> None:
        """
        Parse and chunk the document, embed, and upsert to Qdrant.
        """
        elements = self._parse_and_chunk(file_path, doc_id)

        # Upsert in batches
        for batch in batch_iter(elements, batch_size):
            points = []
            for el in batch:
                vectors = self._compute_vectors_for_element(el)
                payload = self._element_payload(el)

                point_kwargs = {
                    "id": str(uuid.uuid4()),
                    "vector": vectors if vectors else None,
                    "payload": payload,
                }

                points.append(qmodels.PointStruct(**point_kwargs))

            self.qdrant.upsert(
                collection_name=self.collection,
                wait=True,
                points=points,
            )

    def query(
        self,
        q: str,
        k: int = 5,
        doc_id: Optional[str] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Unified multi-modal retrieval:
        1. Always search text and image vectors.
        2. Fuse results using Reciprocal Rank Fusion (RRF).
        3. Rerank the diverse fused candidates (text, tables, images) together.
        """
        ## 1. create enbeddings
        query_text_vec = self._embed_query(q)
        query_image_vec = self._embed_query_for_image(q)

        # define a filter
        flt = None
        if doc_id:
            flt = qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
            )

        # always perform both dense text and dense image search
        candidate_k = max(k * 5, 20)

        dense_res = self.qdrant.search(
            collection_name=self.collection,
            query_vector={"name": "text_dense", "vector": query_text_vec},
            limit=candidate_k,
            query_filter=flt,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        image_res = self.qdrant.search(
            collection_name=self.collection,
            query_vector={"name": "image_dense", "vector": query_image_vec},
            limit=candidate_k,
            query_filter=flt,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        ## 2. Fuse results with RRF
        # rrf k is set to 60 -> referred in the paper
        fused_candidates = self._reciprocal_rank_fusion([dense_res, image_res], k=60)

        ## 3. unified ranking from mixed -> paras, images and tables
        final_candidates = []
        if fused_candidates:
            rerank_pairs = []
            for point in fused_candidates:
                payload = point.payload
                # Create a slightly better text representation for reranking
                text_to_rerank = ""
                element_type = payload.get("element_type")

                if element_type == "table" and payload.get("table_html"):
                    text_to_rerank = f"Table with columns: {html_to_text(payload['table_html'])}"
                elif element_type == "figure":
                    text_to_rerank = f"Image caption: {payload.get('caption', 'No caption')}"
                else:
                    text_to_rerank = payload.get("text") or ""
                
                rerank_pairs.append((q, text_to_rerank))

            # perform reranking on all candidate types at once
            if rerank_pairs:
                scores = self.reranker.predict(rerank_pairs)
                for i, point in enumerate(fused_candidates):
                    point.payload["rerank_score"] = float(scores[i])

            #visual boost by keyword match
            visual_words = ("image", "images", "figure", "fig.", "diagram", "photo", "chart", "graph", "screenshot", "plot")
            has_visual_intent = any(w in q.lower() for w in visual_words)

            if has_visual_intent:
                for point in fused_candidates:
                    if point.payload.get("element_type") == "figure":
                        # apply a proportional boost to images for visual queries
                        point.payload["rerank_score"] *= 1.5 
            
            # sort candidates by the final, potentially boosted, rerank score
            fused_candidates.sort(key=lambda x: x.payload.get("rerank_score", 0.0), reverse=True)
            final_candidates = fused_candidates


        ## 4. build context and return top k
        final_hits = []
        for point in final_candidates[:k]:
            final_hits.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
                "rerank_score": point.payload.get("rerank_score")
            })

        enriched = self._build_contexts(final_hits)
        return enriched

    
    def _fuse_results(self, batch_results: List[List[qmodels.ScoredPoint]]) -> List[Dict[str, Any]]:
        """
        Merge multiple search lists (dense, sparse, etc.) by max score per point id.
        """
        best: Dict[str, Dict[str, Any]] = {}
        for lst in batch_results:
            for sp in lst:
                pid = sp.id
                if pid not in best or sp.score > best[pid]["score"]:
                    best[pid] = {
                        "id": pid,
                        "score": float(sp.score),
                        "payload": sp.payload,
                        "vector": sp.vector,
                    }
        return list(best.values())

    def _build_contexts(self, hits: List[Dict[str, Any]], radius: int = 1) -> List[Dict[str, Any]]:
        """
        Expand each hit with ±radius siblings on same page/doc using payload.order.
        """
        out = []
        for h in hits:
            pl = h["payload"]
            doc_id = pl["doc_id"]
            page = pl["page"]
            order = pl.get("order", None)

            siblings = []
            if order is not None:
                low = max(order - radius, 0)
                high = order + radius
                sib = self.qdrant.scroll(
                    collection_name=self.collection,
                    scroll_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id)),
                            qmodels.FieldCondition(key="page", match=qmodels.MatchValue(value=page)),
                            qmodels.FieldCondition(key="order", range=qmodels.Range(gte=low, lte=high)),
                        ]
                    ),
                    with_payload=True,
                    limit=10,
                )
                siblings = sib[0] if sib and len(sib) > 0 else []

            context_parts = []
    
            if pl.get("element_type") == "table" and pl.get("table_html"):
                context_parts.append("[TABLE CONTENT AVAILABLE]")
                context_parts.append(html_to_text(pl["table_html"])[:2000])
            if pl.get("text"):
                context_parts.append(pl["text"])

            for s in siblings:
                if s.id == h["id"]: continue
                sp = s.payload
                if sp.get("element_type") == "table" and sp.get("table_html"):
                    context_parts.append(html_to_text(sp["table_html"])[:1000])
                if sp.get("text"):
                    context_parts.append(sp["text"])

            context = "\n".join([c for c in context_parts if c]).strip()
            
            out.append({
                "doc_id": doc_id,
                "page": page,
                "element_type": pl.get("element_type"),
                "context": context[:5000],
                "score": h["score"],
                "rerank_score": h.get("rerank_score"),
                "bbox": pl.get("bbox"),
                "citation": {"doc_id": doc_id, "page": page, "order": order},
                "image_path": pl.get("image_path"),
                "image_thumb_b64": pl.get("image_thumb_b64"),
                "table_html": pl.get("table_html"),
            })
        return out

    def _reciprocal_rank_fusion(
        self,
        results_lists: List[List[qmodels.ScoredPoint]],
        k: int = 60
    ) -> List[qmodels.ScoredPoint]:
        """
        Performs Reciprocal Rank Fusion on multiple lists of search results.
        """

        # use a defaultdict to store rrf scores for each unique point ID
        rrf_scores = defaultdict(float)

        # keep track of the full ScoredPoint object for each ID
        all_points = {}

        # iterate through each list of search results
        for results in results_lists:
            for rank, point in enumerate(results):
                if point.id not in all_points:
                    all_points[point.id] = point
                # add the rrf score (1 / (rank + k))
                rrf_scores[point.id] += 1.0 / (rank + k)

        # sort the point IDs by their combined rrf score in descending order
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # build the final, fused list of ScoredPoint objects
        fused_results = [all_points[pid] for pid in sorted_ids]

        return fused_results


## using this file ##
if __name__ == "__main__":

    dp = DocumentProcessing(collection="docs_multimodal")

    # Ingest
    dp.ingest("data/KB_document1.pdf", doc_id="doc-001")

    # Query
    results = dp.query("Show me a diagrams in this document", k=20, doc_id="doc-001")
    print(json.dumps(results, indent=2, ensure_ascii=False))