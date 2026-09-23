# kdsg-rag — an auditable RAG assistant for the revised Bern data protection law

Work in progress. A question-answering assistant over the revised Kantonales
Datenschutzgesetz (KDSG) and related texts, built with retrieval-augmented
generation and documented the way an auditor would examine an AI system:
data provenance, measured answer quality, robustness and traceability.

## Status
- [x] Step 1: project set-up
- [ ] Step 2: corpus and provenance
- [ ] Step 3: ingestion
- [ ] Step 4: baseline RAG
- [ ] Step 5: evaluation

## Structure
```
data/raw/        official source documents + sources.yaml (provenance)
data/processed/  cleaned chunks (generated, not committed)
src/             pipeline code
eval/            test questions and evaluation runs
tests/           pytest
```

## Data Sourcess

All source documents are official legal texts of the Canton of Bern and the Swiss Confederation. Under Art. 5 of the Swiss Copyright Act (URG), laws, ordinances and official decisions are not protected by copyright, so the files are included in this repository unchanged; their origin, version and checksum are recorded in data/raw/sources.yaml.