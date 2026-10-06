"""Per-file statistics of data/processed/chunks.jsonl (run after `python src/ingest.py --dry-run`).

Columns: chunks | articles = distinct article starts (section/article, part 0) | extra = surplus
starts of the same pair (sum of occurrences - 1) | no-article units (part 0) | continuation windows
(part > 0) | text length min/median/max.  Identity: chunks = articles + extra + no-article + continuations.
Then the sections found and the pairs started more than once (genuine duplicates, e.g. a
cross-reference line beginning with "Art. N"); continuations are not duplicates.
"""
import json, collections, pathlib

rows = [json.loads(l) for l in pathlib.Path("data/processed/chunks.jsonl").read_text().splitlines()]
by_file = collections.defaultdict(list)
for r in rows:
    by_file[r["source_file"]].append(r)

for f, rs in by_file.items():
    keys = [f"{r.get('section') or 'main'}/{r['article']}" for r in rs if r["article"] and r["part"] == 0]
    extra = len(keys) - len(set(keys))
    lens = [len(r["text"]) for r in rs]
    sections = sorted({r["section"] for r in rs if r.get("section")})
    print(f"\n{f}: {len(rs)} chunks | articles {len(set(keys))} | extra {extra} | "
          f"no-article {sum(1 for r in rs if not r['article'] and r['part'] == 0)} | "
          f"continuations {sum(1 for r in rs if r['part'] > 0)} | "
          f"len min/median/max {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    print("  sections:", sections or "-")
    dup = [k for k, c in collections.Counter(keys).items() if c > 1]
    print(f"  repeated (section/article): {len(dup)} -> {dup or '-'}")

ids = [r["id"] for r in rows]
print(f"\nids unique: {len(ids) == len(set(ids))} ({len(ids)} chunks total)")