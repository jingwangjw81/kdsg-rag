"""
Step 3 — ingestion: PDF -> section/article-level chunks with metadata -> embeddings -> Chroma.

Usage:
    python src/ingest.py            # ingest everything listed in data/raw/sources.yaml
    python src/ingest.py --dry-run  # only extract + chunk, write chunks.jsonl, no embeddings

Outputs:
    data/processed/chunks.jsonl     one JSON line per chunk (inspect this!)
    data/chroma/                    persistent vector store (collection "kdsg")
"""
import argparse, json, pathlib, re, sys
from collections import Counter
import yaml
import pymupdf

RAW = pathlib.Path("data/raw")
PROCESSED = pathlib.Path("data/processed")
CHROMA_DIR = pathlib.Path("data/chroma")
COLLECTION = "kdsg"
EMBED_MODEL = "intfloat/multilingual-e5-small"   # swap for "BAAI/bge-m3" later if retrieval quality is short

MIN_CHUNK_CHARS = 80          # anything shorter is a fragment (cross-reference line, history entry), not a provision
WINDOW_SIZE, WINDOW_OVERLAP = 1800, 200

# An article heading at line start: "Art. 21", "Art. 21a", "Art.  5".
ART_RE = re.compile(r"^\s*Art\.\s*(\d+[a-z]?)\b", re.MULTILINE)

# Section markers at line start:
#  - ICSGW annex headings: "Anhang 2 zur ICSGW: ...", "Anhang 4 ICSGW"
#  - amendment blocks in any act: "Der Erlass 861.112 Verordnung über ... wird wie folgt geändert:"
SECTION_RE = re.compile(
    r"^\s*(?:Anhang\s+(?P<annex>\d+)\s+(?:zur?\s+)?ICSGW\b.*"
    r"|Der\s+Erlass\s+(?P<amend>\d[\d.]*)\s.*)$",
    re.MULTILINE)

# Historical BELEX PDFs end with amendment-history sections whose headings stand alone on a line,
# e.g. "Änderungstabelle", "Chronologische Übersicht", or "Tabelle der Änderungen".
# We only trim from the final such heading if it appears in the second half of the document,
# so ordinary earlier references or title-page notes do not trigger a false cut.
HISTORY_RE = re.compile(
    r"^\s*(?:Änderungstabelle[n]?|Chronologische Übersicht|Tabelle der Änderungen)"
    r"(?:\s*[-–—]\s*.*)?$",
    re.MULTILINE,
)

# Known footer/header strings that survive the repetition filter (seen in the ICSGW extraction).
KNOWN_BOILERPLATE = ("Beschluss mit Anhang 5 und 6",)


def extract_text(pdf_path: pathlib.Path) -> str:
    """Concatenate page text; drop page headers/footers that repeat on most pages or are known boilerplate."""
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    line_counts = Counter(l.strip() for p in pages for l in p.splitlines() if l.strip())
    n_pages = max(len(pages), 1)
    boiler = {l for l, c in line_counts.items() if c >= max(3, 0.6 * n_pages)}
    boiler.update(KNOWN_BOILERPLATE)
    cleaned = ["\n".join(l for l in p.splitlines() if l.strip() not in boiler) for p in pages]
    return "\n".join(cleaned)


def cut_history(text: str) -> str:
    """Truncate at the last amendment-history heading that stands alone on a line and lies in the
    second half of the text (BELEX prints a footnote 'Änderungstabellen am Schluss des Erlasses'
    near the title, which must not trigger the cut)."""
    cut = len(text)
    for m in HISTORY_RE.finditer(text):
        if m.start() > len(text) * 0.5:
            cut = m.start()
    return text[:cut]


def split_sections(text: str):
    """Yield (section_name or None, section_text). Section marker lines are removed from the text,
    which also strips the running page headers that repeat the annex title."""
    pos, section, buf = 0, None, []
    for m in SECTION_RE.finditer(text):
        buf.append(text[pos:m.start()])
        name = f"Anhang {m.group('annex')}" if m.group("annex") else f"Änderung {m.group('amend')}"
        if name != section:
            yield section, "".join(buf)
            section, buf = name, []
        pos = m.start() if m.group("amend") else m.end()   # keep amendment line, drop annex marker
    buf.append(text[pos:])
    yield section, "".join(buf)


def split_articles(text: str):
    """Yield (section, article_no or None, chunk_text). Text before the first article of a section
    becomes a 'preamble' chunk; sections without articles become one chunk (windowed later)."""
    for section, sec_text in split_sections(text):
        matches = list(ART_RE.finditer(sec_text))
        if not matches:
            if sec_text.strip():
                yield section, None, sec_text.strip()
            continue
        if matches[0].start() > 0:
            pre = sec_text[: matches[0].start()].strip()
            if pre:
                yield section, None, pre
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(sec_text)
            yield section, m.group(1), sec_text[m.start():end].strip()


def window(text: str, size: int = WINDOW_SIZE, overlap: int = WINDOW_OVERLAP):
    """Fallback for over-long pieces: fixed windows with overlap (character-based)."""
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
        if s.get("role") == "historical":
            text = cut_history(text)
        stem = pathlib.Path(s["file"]).stem
        n = dropped = 0
        seen = Counter()                     # (section, article) -> kept occurrences, to keep ids unique
        for section, art, body in split_articles(text):
            sec = (section or "main").replace(" ", "")
            pieces = [p for p in window(body) if len(p.strip()) >= MIN_CHUNK_CHARS]
            dropped += sum(1 for _ in window(body)) - len(pieces)
            if not pieces:
                continue
            seen[(sec, art)] += 1
            suffix = f"_{seen[(sec, art)]}" if seen[(sec, art)] > 1 else ""
            for j, piece in enumerate(pieces):
                chunks.append({
                    "id": f"{stem}__{sec}__art{art or 'pre'}{suffix}__{j}",
                    "text": piece,
                    "source_file": s["file"],
                    "title": s.get("title"),
                    "legal_ref": s.get("legal_ref"),
                    "role": s.get("role"),
                    "status": s.get("status"),
                    "valid_from": str(s.get("valid_from")),
                    "valid_until": str(s.get("valid_until")),
                    "section": section,
                    "article": art,
                    "part": j,
                })
                n += 1
        print(f"{s['file']}: {n} chunks ({dropped} fragments dropped)")
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
    try:                                    # start fresh so re-ingestion never leaves stale chunks behind
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