"""
Step 3 — ingestion: PDF -> section/chapter/article-level chunks with metadata -> embeddings -> Chroma.

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

# A table-of-contents heading at line start; the TOC lists "Art. N" lines that would otherwise be
# split as articles (seen in ICSGW Anhang 6).
TOC_RE = re.compile(r"^\s*Inhalt(?:sverzeichnis)?\s*$", re.MULTILINE)

# chapter headings at line start, e.g. "3.2 Register- und Verzeichnispflicht" or "1. Gegenstand".
# Paragraph numbers ("1 Die Behörde darf ...") also start with a digit; find_chapters() excludes them by
# (a) sentence punctuation at the end, (b) a leading article/pronoun, (c) what follows the line.
CHAPTER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-ZÄÖÜ][^\n]{2,80}?)\s*$")
NOT_HEADING_START = re.compile(r"^(Die|Der|Das|Dies\w*|Sie|Er|Es|In|Im|Für|Bei|Folgende|Wenn|Wer|Aufgehoben)\b")
PARA_RE = re.compile(r"^\s*\d+\s+\S")          # a paragraph line: digit, space, text

# Section markers at line start:
#  - ICSGW annex headings: "Anhang 2 zur ICSGW: ...", "Anhang 4 ICSGW"
#  - amendment blocks in any act: "Der Erlass 861.112 Verordnung über ... wird wie folgt geändert:"
SECTION_RE = re.compile(
    r"^\s*(?:Anhang\s+(?P<annex>\d+)\s+(?:zur?\s+)?ICSGW\b.*"
    r"|Der\s+Erlass\s+(?P<amend>\d[\d.]*)\s.*)$",
    re.MULTILINE)

# Historical BELEX PDFs end with amendment-history tables whose headings stand alone on a line,
# e.g. "Änderungstabelle - nach Beschluss" / "- nach Artikel". We trim from the first such heading
# in the second half of the document, so the title-page footnote does not trigger a false cut.
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
    """Truncate at the first amendment-history heading in the second half of the text.
    BELEX historical PDFs end with the tables 'Änderungstabelle - nach Beschluss' and
    '... - nach Artikel'; cutting at the first removes both. The 50% guard skips the
    footnote 'Änderungstabellen am Schluss des Erlasses' printed near the title."""
    for m in HISTORY_RE.finditer(text):
        if m.start() > len(text) * 0.5:
            return text[:m.start()]
    return text


def cut_toc(text: str) -> str:
    """Remove a table of contents: from the 'Inhaltsverzeichnis' line up to the second occurrence
    of the first listed article number, which is where the real text begins."""
    m = TOC_RE.search(text)
    if not m:
        return text
    arts = list(ART_RE.finditer(text, m.end()))
    if len(arts) < 2:
        return text
    first_no = arts[0].group(1)
    for a in arts[1:]:
        if a.group(1) == first_no:
            return text[:m.start()] + text[a.start():]
    return text



def _heading_title(line: str):
    """'3.2 Register- und Verzeichnispflicht' -> ('3.2', 'Register- und Verzeichnispflicht'), else None."""
    m = CHAPTER_RE.match(line)
    if not m:
        return None
    title = m.group(2).strip()
    if title[-1] in ".:,;" or NOT_HEADING_START.match(title):
        return None
    return m.group(1), title



def find_chapters(text: str):
    """Return [(char_offset, 'number title'), ...] for lines that look like chapter headings.
    A candidate (CHAPTER_RE, no sentence punctuation, no leading article/pronoun) is accepted only
    if one of the next two lines is an article line or another heading candidate; in texts without
    any 'Art.' lines (ICSGW annexes 1-4) a paragraph line also counts as confirmation."""
    lines = text.split("\n")
    has_articles = bool(ART_RE.search(text))
    offsets, pos = [], 0
    for l in lines:
        offsets.append(pos)
        pos += len(l) + 1
    out = []
    for i, l in enumerate(lines):
        h = _heading_title(l)
        if not h:
            continue
        following = lines[i + 1:i + 3]
        confirmed = any(ART_RE.match(x) or _heading_title(x) for x in following)
        if not confirmed and not has_articles:
            confirmed = any(PARA_RE.match(x) for x in following)
        if confirmed:
            out.append((offsets[i], f"{h[0]} {h[1]}"))
    return out



def chapter_at(chapters, pos: int) -> str:
    """Title of the last chapter heading that starts at or before pos ('' if none)."""
    current = ""
    for p, title in chapters:
        if p <= pos:
            current = title
        else:
            break
    return current


def split_sections(text: str):
    """Yield (section_name or None, section_text), cutting at each SECTION_RE match.
    Annex headings ("Anhang 2 zur ICSGW") are dropped from the text; repeats of the same
    heading (running page headers) do not open a new section and are dropped too.
    Amendment lines ("Der Erlass 861.112 ... wird wie folgt geändert") are kept, as they
    name the amended ordinance. Text before the first marker has section None."""
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
    """Yield (section, chapter, article_no or None, chunk_text). Text before the first article of a
    section becomes a 'preamble' chunk; sections without articles become one chunk (windowed later).
    The chapter is the last chapter heading before the chunk starts ('' if none)."""      
    for section, sec_text in split_sections(text):
        sec_text = cut_toc(sec_text)
        chapters = find_chapters(sec_text)                                            
        matches = list(ART_RE.finditer(sec_text))
        if not matches:
            if sec_text.strip():
                yield section, chapter_at(chapters, 0), None, sec_text.strip()        
            continue
        if matches[0].start() > 0:
            pre = sec_text[: matches[0].start()].strip()
            if pre:
                yield section, chapter_at(chapters, 0), None, pre                     
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(sec_text)
            yield section, chapter_at(chapters, m.start()), m.group(1), sec_text[m.start():end].strip()  


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
        for section, chapter, art, body in split_articles(text):                      
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
                    "legal_ref": s.get("legal_ref") or "",
                    "role": s.get("role") or "",
                    "status": s.get("status") or "",
                    "valid_from": str(s.get("valid_from") or ""),
                    "valid_until": str(s.get("valid_until") or ""),
                    "section": section or "",
                    "chapter": chapter or "",                                          
                    "article": art or "",
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


def embed_text(c: dict) -> str:
    """Text that is embedded: act / section / chapter as a context prefix, then the chunk text.
    The stored text (chunks.jsonl, citations) stays the plain chunk; only the vector sees the prefix."""
    prefix = " / ".join(x for x in (c["title"], c["section"], c["chapter"]) if x)
    return f"[{prefix}]\n{c['text']}"


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

    nodes = [TextNode(id_=c["id"], text=embed_text(c),                                 
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