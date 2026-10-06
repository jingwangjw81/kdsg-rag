# kdsg-rag — an auditable RAG assistant for the revised Bern data protection law

Work in progress. A question-answering assistant over the revised Kantonales
Datenschutzgesetz (KDSG) and related texts, built with retrieval-augmented
generation and documented the way an auditor would examine an AI system:
data provenance, measured answer quality, robustness and traceability.

## Status
- [x] Step 1: project set-up
- [x] Step 2: corpus and provenance
- [x] Step 3: ingestion
- [ ] Step 4: baseline RAG
- [ ] Step 5: evaluation

## Structure
```
data/raw/        official source documents + sources.yaml (provenance)
data/processed/  cleaned chunks (generated, not committed)
src/             pipeline code (checksums, ingest, inspect_chunks, smoke_test)
eval/            test questions and evaluation runs
tests/           pytest
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
`src/ingest.py` turns the PDFs into chunks and a Chroma index:
- Text extraction with pymupdf; repeated page headers/footers removed.
- Historical versions: the amendment-history tables at the end are cut off.
- Sections: ICSGW annexes and "Der Erlass … wird wie folgt geändert" amendment blocks
  become their own sections, so article numbers are unique per section.
- Chunking: one chunk per article (`Art. N` at line start) within its section; a table of
  contents listing articles is removed; articles longer than 1,800 characters are split
  into overlapping windows (200 characters); pieces under 80 characters are dropped.
- Metadata per chunk: source file, act, legal reference, role, status, validity dates,
  section, article, window index. Chunk ids are unique (`tests/test_ingest.py`).
- Embeddings: `intfloat/multilingual-e5-small` (local, CPU); vector store: Chroma,
  collection `kdsg`.

Chunk counts (`python src/inspect_chunks.py`): 
KDSG new 68, 
KDSG old 43, 
KDSV 23,
DSV old 19, 
ICSG 43, 
IDSV 53, 
DSG 81, 
ICSGW 93, 
AGB 10; 
433 chunks in total.

## Limitations
- The ICSGW annexes 1–4 and the AGB ISDS BE are laid out as tables; clause numbers and
  titles are not recoverable from plain-text extraction, so these sections are indexed as
  1,800-character windows rather than clause-level chunks. Citations into them name the
  section and window, not the clause.
- The ICSG and IDSV files are the adopted texts before publication in the BSG
  (numbers pending); they are to be replaced by the published versions after 2026-11-01.
- The assistant answers from the versions listed in `sources.yaml` and knows nothing
  about later changes.

## Reproducibility
Dependencies are declared in `requirements.txt`; exact versions that passed the smoke
test are in `requirements.lock`. Source versions are pinned in `sources.yaml`; model
names are recorded in `src/ingest.py` and, from step 5, with each evaluation run.
