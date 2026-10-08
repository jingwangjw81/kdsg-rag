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
CHAPTER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-ZÄÖÜ][^\n]{2,60}?)\s*$")
NOT_HEADING_START = re.compile(r"^(Die|Der|Das|Dies\w*|Sie|Er|Es|In|Im|Für|Bei|Folgende|Wenn|Wer|Aufgehoben)\b")
# sentences, not headings: finite verbs, footnote citations ("DVG; BSG 109.1"), wrapped lines ending in a hyphen
SENTENCE_HINT = re.compile(r"\b(müssen|dürfen|ist|sind|kann|können|wird|werden|gilt|gelten|hat|haben|darf|soll|sollen"
                           r"|besteht|bestehen|erfolgt|nach Absatz)\b|;|\b(BSG|SR)\s+\d")
PARA_RE = re.compile(r"^\s*\d+\s+\S")          # a paragraph line: digit, space, text
# Fedlex style headings: "1. Kapitel: Begriffe" (depth 0), "3. Abschnitt: Amtshilfe" (depth 1)
FEDLEX_LEVEL = {"Kapitel:": 0, "Abschnitt:": 1}

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

# A line that is only a number ("1.", "3.2", "5.1.1") or an article label ("Art. 7"): in table-cell layouts
# (ICSGW Anhang 6, AGB ISDS BE) the title follows on the next line and must be joined back.
NUM_ONLY_RE = re.compile(r"^\s*(\d+(?:\.\d+)*\.?|Art\.\s*\d+[a-z]?)\s*$")

# Known footer/header strings that survive the repetition filter (seen in the ICSGW extraction).
KNOWN_BOILERPLATE = ("Beschluss mit Anhang 5 und 6",)

# A line that is only a heading number ("1.", "5.1.1") or an article number ("Art. 3"); in
# table-cell layouts (ICSGW Anhang 6, AGB ISDS BE) the title follows on the next line.
NUM_ONLY_RE = re.compile(r"^\s*(\d+(?:\.\d+)*\.?|Art\.\s*\d+[a-z]?)\s*$")


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


def join_split_headings(text: str) -> str:
    """'1.' + newline + 'Allgemeine Bestimmungen' -> '1. Allgemeine Bestimmungen', and
    'Art. 1' + newline + 'Gegenstand' -> 'Art. 1 Gegenstand' (table-cell layout). A number-only
    line followed by another number-only line or a blank line is left as it is."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        cur = lines[i]
        if (NUM_ONLY_RE.match(cur) and i + 1 < len(lines)
                and lines[i + 1].strip() and not NUM_ONLY_RE.match(lines[i + 1])):
            out.append(f"{cur.strip()} {lines[i + 1].strip()}")
            i += 2
        else:
            out.append(cur)
            i += 1
    return "\n".join(out)


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
    num, title = m.group(1), m.group(2).strip()
    if "." not in num and len(num) >= 4:                       # "1000 Franken ..." is a table row, not a heading
        return None
    if title[-1] in ".:,;-\u00ad" or NOT_HEADING_START.match(title) or SENTENCE_HINT.search(title):
        return None
    return num, title


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
    out, expected = [], 1                      # 'expected': next top-level number in article-free sections
    for i, l in enumerate(lines):
        h = _heading_title(l)
        if not h:
            continue
        following = lines[i + 1:i + 3]
        confirmed = any(ART_RE.match(x) or _heading_title(x) for x in following)
        # A chapter heading sits between articles; a numbered list item inside an article
        # ("1. Cipher Suiten" in a glossary) is preceded by article text. For such candidates the
        # confirming article/heading must be on the very next line, which a list item's
        # definition text never satisfies.                                              
        prev = next((x for x in reversed(lines[:i]) if x.strip()), "")
        structural = (not prev or _heading_title(prev) or ART_RE.match(prev) or NUM_ONLY_RE.match(prev)
                      or SECTION_RE.match(prev) or re.match(r"^\s*\d+/\d+\s*$", prev))
        if has_articles and not structural:
            nxt = next((x for x in lines[i + 1:i + 4] if x.strip()), "")   # next non-blank line
            confirmed = bool(ART_RE.match(nxt) or _heading_title(nxt))
        if not has_articles:
            # tables in article-free annexes have numbers in the first column; accept only headings
            # whose top-level number continues the sequence 1, 2, 3 ...
            top = int(h[0].split(".")[0])
            if top != expected and "." not in h[0]:
                continue
            if "." not in h[0]:
                expected = top + 1
            confirmed = confirmed or any(PARA_RE.match(x) for x in following)
        if confirmed:
            out.append((offsets[i], f"{h[0]} {h[1]}"))
    return out


def _depth(title: str) -> int:
    for word, d in FEDLEX_LEVEL.items():
        if word in title:
            return d
    return title.split(" ", 1)[0].count(".")


def chapter_at(chapters, pos: int) -> str:
    """Full heading path for pos, e.g. '2 Bearbeitung von Personendaten / 2.1 Grundsätze' ('' if none).
    Depth comes from the numbering ("2.1" is under "2") or, in Fedlex texts, from the words
    "Kapitel:" (top) and "Abschnitt:" (below)."""
    stack = []
    for p, title in chapters:
        if p > pos:
            break
        d = _depth(title)
        stack = stack[:d] + [title]
    return " / ".join(stack)


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
         # join table-cell headings only now, after the history cut and the section split, so that
         # "Änderungstabelle" headings and "Der Erlass ..." markers are still at line start when matched
        sec_text = cut_toc(join_split_headings(sec_text))
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
