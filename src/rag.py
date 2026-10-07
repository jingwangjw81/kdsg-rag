"""
Step 4 — baseline RAG (version 0.1): question -> retrieve chunks from Chroma -> answer with an LLM.

Usage:
    python src/rag.py "Welche Behörden müssen ein Register der Datensammlungen führen?"
    python src/rag.py --provider mistral --top-k 6 "Was hat sich bei der Registerpflicht geändert?"
    python src/rag.py --show-context "..."          # also print the retrieved passages

Design decisions (all measurable later):
    - retrieval: top-k nearest chunks by embedding similarity, same model as ingestion
    - historical versions are excluded unless the question is about changes (--include-historical
      or trigger words such as "geändert", "alt", "bisher", "Unterschied")
    - the LLM answers in German ONLY from the retrieved passages, cites them, and says explicitly
      when the passages do not contain the answer (refusal is a feature, not a failure)
"""
import argparse, json, os, re, sys, time
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.ingest import CHROMA_DIR, COLLECTION, EMBED_MODEL

VERSION = "0.1"
DEFAULT_TOP_K = 5
MODELS = {"gemini": "gemini-2.5-flash", "mistral": "mistral-small-latest"}

CHANGE_WORDS = re.compile(r"\b(geändert|Änderung|neu\b|alt\b|alte[nrs]?\b|bisher|früher|vorher|Unterschied|revidiert|Revision)",
                          re.IGNORECASE)

SYSTEM_PROMPT = """Du bist ein sorgfältiger Assistent für das Datenschutz- und Informationssicherheitsrecht des Kantons Bern.
Beantworte die Frage AUSSCHLIESSLICH anhand der nummerierten Auszüge unten. Regeln:
1. Verwende keine Informationen, die nicht in den Auszügen stehen, auch wenn du sie zu kennen glaubst.
2. Belege jede Aussage mit der Quelle in eckigen Klammern, z.B. [1] oder [1, 3].
3. Nenne Erlass und Artikel, wenn sie im Auszug erkennbar sind (z.B. "Art. 21 KDSG").
4. Wenn die Auszüge die Frage nicht oder nur teilweise beantworten, sage das ausdrücklich
   ("Die vorliegenden Auszüge enthalten dazu keine Angaben.") statt zu raten.
5. Antworte auf Deutsch, knapp und präzise."""


def wants_history(question: str) -> bool:
    return bool(CHANGE_WORDS.search(question))


def build_filter(include_historical: bool):
    """Chroma 'where' filter. Historical versions are kept out unless requested."""
    return None if include_historical else {"role": {"$ne": "historical"}}


def retrieve(question: str, top_k: int, include_historical: bool, embed: HuggingFaceEmbedding):
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    qvec = embed.get_query_embedding(question)
    res = col.query(query_embeddings=[qvec], n_results=top_k, where=build_filter(include_historical),
                    include=["documents", "metadatas", "distances"])
    hits = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"id": cid, "text": doc, "distance": round(float(dist), 4),
                     "title": meta.get("title", ""), "legal_ref": meta.get("legal_ref", ""),
                     "section": meta.get("section", ""), "article": meta.get("article", ""),
                     "role": meta.get("role", ""), "status": meta.get("status", ""),
                     "valid_from": meta.get("valid_from", ""), "valid_until": meta.get("valid_until", "")})
    return hits


def label(h: dict) -> str:
    """Human-readable source label, e.g. 'KDSG, BSG 152.04, Art. 21 (in force)'."""
    parts = [h["title"]]
    if h["legal_ref"]:
        parts.append(h["legal_ref"])
    if h["section"]:
        parts.append(h["section"])
    if h["article"]:
        parts.append(f"Art. {h['article']}")
    if h["status"]:
        parts.append(f"({h['status']})")
    return ", ".join(parts)


def build_prompt(question: str, hits: list) -> str:
    context = "\n\n".join(f"[{i}] {label(h)}\n{h['text']}" for i, h in enumerate(hits, 1))
    return f"{SYSTEM_PROMPT}\n\nAUSZÜGE:\n{context}\n\nFRAGE: {question}\n\nANTWORT:"


def make_llm(provider: str):
    if provider == "gemini":
        from llama_index.llms.google_genai import GoogleGenAI
        return GoogleGenAI(model=MODELS["gemini"], temperature=0)
    if provider == "mistral":
        from llama_index.llms.mistralai import MistralAI
        return MistralAI(model=MODELS["mistral"], temperature=0)
    raise ValueError(f"unknown provider {provider}")


def answer(question: str, provider: str = "gemini", top_k: int = DEFAULT_TOP_K,
           include_historical=None, embed=None) -> dict:
    """Run the full pipeline and return a dict with everything needed for logging and evaluation."""
    if include_historical is None:
        include_historical = wants_history(question)
    embed = embed or HuggingFaceEmbedding(model_name=EMBED_MODEL)
    t0 = time.time()
    hits = retrieve(question, top_k, include_historical, embed)
    prompt = build_prompt(question, hits)
    text = str(make_llm(provider).complete(prompt)).strip()
    return {
        "version": VERSION, "provider": provider, "model": MODELS[provider],
        "embed_model": EMBED_MODEL, "top_k": top_k, "include_historical": include_historical,
        "question": question, "answer": text,
        "sources": [{k: h[k] for k in ("id", "distance", "title", "legal_ref", "section", "article", "status")} for h in hits],
        "latency_s": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--provider", choices=list(MODELS), default="gemini")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--include-historical", action="store_true", help="force inclusion of superseded versions")
    ap.add_argument("--show-context", action="store_true", help="print the retrieved passages")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args()

    embed = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    inc = True if args.include_historical else None
    result = answer(args.question, args.provider, args.top_k, inc, embed)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit()
    if args.show_context:
        hits = retrieve(args.question, args.top_k, result["include_historical"], embed)
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {label(h)}  (distance {h['distance']})\n{h['text'][:400]}\n")
    print(f"--- {result['provider']} / {result['model']} | top_k={result['top_k']} | "
          f"historical={'yes' if result['include_historical'] else 'no'} | {result['latency_s']}s ---")
    print(result["answer"])
    print("\nQuellen:")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {label({**s, 'valid_from': '', 'valid_until': ''})}")
