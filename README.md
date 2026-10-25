# kdsg-rag — an auditable RAG assistant for the revised Bern data protection law

A question-answering assistant over the revised Kantonales Datenschutzgesetz (KDSG) of the Canton of Bern and related texts, built with retrieval-augmented generation and documented the way an auditor would examine an AI system: data provenance, measured answer quality, robustness and traceability. Work in progress; the baseline system works end to end.

## What it does today
- Ask a question in German, then the assistant answers only from the indexed legal texts, cites every statement with `[n]`, and names act, legal reference, chapter, article and legal status for each source. 
- Superseded versions are excluded from retrieval unless the question is about what changed. 
- Questions outside the corpus are refused rather than guessed. 

## Status
| Step | Content | State |
|---|---|---|
| 1 | Project set-up (Codespaces, dependencies, smoke test of both LLM providers) | done
| 2 | Corpus and provenance (`sources.yaml`, checksums, tests) | done
| 3 | Ingestion: article-level chunking with section and chapter metadata | done
| 4 | Baseline RAG: retrieval with legal-status filter, cited answers, refusal | in progress
| 5 | Evaluation: 30–50 German test questions incl. change and out-of-scope questions; RAGAS metrics (faithfulness, answer relevance, context precision/recall); over-/under-refusal | planned
| 6 | Robustness: paraphrases, adversarial and ambiguous questions | planned
| 7 | Traceability: per-answer log of question, retrieved chunks, model, prompt version | planned
| 8 | Streamlit UI | planned
| 9 | Model comparison (Gemini vs. Mistral) on the evaluation set | planned
| 10 | Audit-style report of the results | planned


## Architecture
```
PDF (data/raw)
  -> extract_text      pymupdf; repeated headers/footers and page numbers deleted
  -> split_sections    annexes and amendment blocks become sections
  -> find_chapters     numbered chapter headings detected and attached to each article
  -> split_articles    one chunk per article; long articles windowed (1,800 / 200 chars)
  -> embed             intfloat/multilingual-e5-small, local CPU; embedding text is prefixed
                       with "[act / section / chapter]" so retrieval sees the context
  -> Chroma            persistent collection "kdsg", full metadata per chunk
Question
  -> retrieve          top-k by vector similarity; metadata filter excludes role=historical
                       unless the question contains change words ("geändert", "neu", "früher" ...)
  -> prompt            German system prompt: answer only from the passages, cite [n],
                       refuse if the passages do not contain the answer
  -> LLM               Gemini (gemini-3.6-flash) or Mistral (codestral-2508; a general Mistral model is to be compared in step 9), temperature 0
  -> answer            text + source list (act, legal reference, chapter, article, status)
```

## Data sources
All source documents are official legal texts of the Canton of Bern and the Swiss
Confederation. Under Art. 5 of the Swiss Copyright Act (URG), laws, ordinances and
official decisions are not protected by copyright, so the files are included in this
repository unchanged. For every file, `data/raw/sources.yaml` records publisher, URL,
legal reference, validity dates, legal status, role in the corpus and SHA-256 checksum;
`tests/test_sources.py` fails if a file is missing or altered.

| File | Act | Role | Status |
|---|---|---|---|
| 1_KDSG_new.pdf | KDSG, BSG 152.04 | primary | in force since 2026-09-01 |
| 2_KDSG_old.pdf | KDSG (previous version) | historical | superseded 2026-08-31 |
| 3_KDSV_new.pdf | KDSV, BSG 152.040.1 | primary | in force since 2026-09-01 |
| 4_DSV_old.pdf | DSV (previous version) | historical | superseded 2026-08-31 |
| 5_ICSG.pdf | ICSG | related | adopted, in force from 2026-11-01 |
| 6_IDSV.pdf | IDSV | related | adopted, in force from 2026-11-01 |
| 7_DSG.pdf | Federal DSG, SR 235.1 | related | in force (consolidated 2025-07-07) |
| 8_ICSGW.pdf | ICSGW (KAIO directive) | guidance | in force, replacement announced |
| 9_AGB_ISDS_BE.pdf | AGB ISDS BE | guidance | in force, replacement announced |

## Data processing
Design rule: extracted text is never rewritten, only deleted. Deletions are limited to
layout (repeated headers/footers, page numbers), amendment-history tables and tables of
contents. Structure (sections, chapters, articles) is detected and stored as metadata; the
chunk text stays as extracted so that every citation can be checked against the PDF.
- Text extraction with pymupdf. Lines repeated on most pages (headers/footers) are dropped.
A line is dropped as a page number only if it is `n/m` or a bare number equal to its own
page index — table values (e.g. key lengths in ICSGW Anhang 6) and footnote markers stay.
- Historical versions: the amendment-history tables at the end are cut off.
- Sections: ICSGW annexes and "Der Erlass … wird wie folgt geändert" amendment blocks
become their own sections, so article numbers are unique per section.
- Chapters: numbered headings (`3`, `3.2`, Fedlex `3. Kapitel:` / `2. Abschnitt:`) are
detected and attached to each article as a path, e.g.
`3 Pflichten der Behörden und von beauftragten Dritten / 3.2 Register- und Verzeichnispflicht`.
A heading is accepted only if an article or another heading follows it, which keeps
paragraph numbers, list items, footnotes and table rows out. The chapter is metadata; the chunk text is not changed.
- Chunking: one chunk per article (`Art. N` at line start) within its section; a table of
contents listing articles is removed; articles longer than 1,800 characters are split
into overlapping windows (200 characters); pieces under 80 characters are dropped.
- Metadata per chunk: source file, act, legal reference, role, status, validity dates,
section, chapter, article, window index. Chunk ids are unique (`tests/test_ingest.py`).
- Embeddings: `intfloat/multilingual-e5-small` (local, CPU); vector store: Chroma,
collection `kdsg`.
Chunk counts (`python src/inspect_chunks.py`): KDSG new 68 · KDSG old 43 · KDSV 23 ·
DSV old 19 · ICSG 43 · IDSV 53 · DSG 81 · ICSGW 93 · AGB 10 — 433 chunks in total.

## Limitations
- The ICSGW annexes and the AGB ISDS BE are laid out as tables; clause numbers and
titles are not recoverable from plain-text extraction, so these sections are indexed as
1,800-character windows rather than clause-level chunks and carry no chapter. Citations
into them name the section and window, not the clause.
- Chapter headings wrapped over two PDF lines are not detected: KDSV
chapter 4 (Art. 12–15) and in the DSG the 1., 5., 6. and 7. Kapitel and two Abschnitte.
The affected articles carry the preceding chapter. Headings on one line are detected regardless
of length. 
- The page-number rule deletes one footnote marker that coincides with page 1 of the DSG;
no article text is affected.
- The ICSG and IDSV files are the adopted texts before publication in the BSG
(numbers pending); they are to be replaced by the published versions after 2026-11-01.
- The assistant answers from the versions listed in `sources.yaml` and knows nothing
about later changes. Answer quality has not yet been measured (step 5).

## Reproducibility
- Dependencies are declared in `requirements.txt`; source versions are pinned in `sources.yaml`; model
names are recorded in `src/ingest.py` and, from step 5, with each evaluation run.
