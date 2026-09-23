"""
Step 3 — ingestion: PDF -> article-level chunks with metadata -> embeddings -> Chroma.

Usage:
    python src/ingest.py            # ingest everything listed in data/raw/sources.yaml
    python src/ingest.py --dry-run  # only extract + chunk, write chunks.jsonl, no embeddings

Outputs:
    data/processed/chunks.jsonl     one JSON line per chunk (inspect this!)
    data/chroma/                    persistent vector store (collection "kdsg")
"""
import argparse, json, pathlib, re, sys
import yaml
import pymupdf

RAW = pathlib.Path("data/raw")
PROCESSED = pathlib.Path("data/processed")
CHROMA_DIR = pathlib.Path("data/chroma")
COLLECTION = "kdsg"
EMBED_MODEL = "intfloat/multilingual-e5-small"   # swap for "BAAI/bge-m3" later if quality is short

# An article heading in Swiss legal texts: "Art. 21", "Art. 21a", "Art.  5" (extra spaces), start of line.
ART_RE = re.compile(r"^\s*Art\.\s*(\d+[a-z]?)\b", re.MULTILINE)


def extract_text(pdf_path: pathlib.Path) -> str:
    """Concatenate page text; drop page headers/footers that repeat on every page."""
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    # remove lines that occur on (almost) every page: typically running headers / page numbers
    from collections import Counter
    line_counts = Counter(l.strip() for p in pages for l in p.splitlines() if l.strip())
    n_pages = max(len(pages), 1)
    boiler = {l for l, c in line_counts.items() if c >= max(3, 0.6 * n_pages)}
    cleaned = []
    for p in pages:
        cleaned.append("\n".join(l for l in p.splitlines() if l.strip() not in boiler))
    return "\n".join(cleaned)


def split_articles(text: str):
    """Yield (article_no or None, chunk_text). Text before the first article becomes a 'preamble' chunk."""
    matches = list(ART_RE.finditer(text))
    if not matches:
        yield None, text
        return
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            yield None, pre
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[m.start():end].strip()
        yield m.group(1), chunk


def window(text: str, size: int = 1800, overlap: int = 200):
    """Fallback for over-long articles: fixed windows with overlap (character-based)."""
    if len(text) <= size:
        yield text
        return
    start = 0
    while start < len(text):
        yield text[start:start + size]
        start += size - overlap


def build_chunks():
    meta = yaml.safe_load((RAW / "sources.yaml").read_text())
    chunks = []
    for s in meta["sources"]:
        pdf = RAW / s["file"]
        if not pdf.exists():
            print(f"skip (missing): {pdf}", file=sys.stderr)
            continue
        text = extract_text(pdf)
        n = 0
        for art, body in split_articles(text):
            for j, piece in enumerate(window(body)):
                chunks.append({
                    "id": f"{pathlib.Path(s['file']).stem}__art{art or 'pre'}__{j}",
                    "text": piece,
                    "source_file": s["file"],
                    "title": s.get("title"),
                    "role": s.get("role"),
                    "status": s.get("status"),
                    "legal_ref": s.get("legal_ref"),
                    "valid_from": str(s.get("valid_from")),
                    "valid_until": str(s.get("valid_until")),
                    "article": art,
                    "part": j,
                })
                n += 1
        print(f"{s['file']}: {n} chunks")
    return chunks


def write_jsonl(chunks):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(chunks)} chunks -> {out}")


def embed_and_store(chunks):
    import chromadb
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # start fresh each run so re-ingestion never leaves stale chunks behind
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.get_or_create_collection(COLLECTION)

    nodes = [TextNode(id_=c["id"], text=c["text"],
                      metadata={k: v for k, v in c.items() if k not in ("id", "text")})
             for c in chunks]
    embed = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    store = ChromaVectorStore(chroma_collection=collection)
    ctx = StorageContext.from_defaults(vector_store=store)
    VectorStoreIndex(nodes, storage_context=ctx, embed_model=embed, show_progress=True)
    print(f"stored {collection.count()} vectors in {CHROMA_DIR}/{COLLECTION} using {EMBED_MODEL}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="extract and chunk only")
    args = ap.parse_args()
    chunks = build_chunks()
    write_jsonl(chunks)
    if not args.dry_run:
        embed_and_store(chunks)
