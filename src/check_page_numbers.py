import pathlib, sys
sys.path.insert(0, "src")
import pymupdf
from ingest import is_page_number
for pdf in sorted(pathlib.Path("data/raw").glob("*.pdf")):
    hits = [(pno, l.strip()) for pno, page in enumerate(pymupdf.open(pdf), 1)
            for l in page.get_text().splitlines() if is_page_number(l, pno)]
    print(pdf.name, len(hits), hits)