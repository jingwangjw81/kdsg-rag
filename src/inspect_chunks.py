import json, collections, pathlib

rows = [json.loads(l) for l in pathlib.Path("data/processed/chunks.jsonl").read_text().splitlines()]
by_file = collections.defaultdict(list)
for r in rows:
    by_file[r["source_file"]].append(r)

for f, rs in by_file.items():
    keys = [f"{r.get('section') or 'main'}/{r['article']}" for r in rs if r["article"]]
    lens = [len(r["text"]) for r in rs]
    sections = sorted({r["section"] for r in rs if r.get("section")})
    print(f"\n{f}: {len(rs)} chunks | articles {len(set(keys))} | "
          f"no-article {sum(1 for r in rs if not r['article'])} | "
          f"continuations {sum(1 for r in rs if r['part'] > 0)} | "
          f"len min/median/max {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    print("  sections:", sections or "-")
    dup = [k for k, c in collections.Counter(keys).items() if c > 1]
    print("  repeated (section/article):", dup[:8] or "-")

ids = [r["id"] for r in rows]
print(f"\nids unique: {len(ids) == len(set(ids))} ({len(ids)} chunks total)")