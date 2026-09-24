import json, collections, pathlib
rows = [json.loads(l) for l in pathlib.Path("data/processed/chunks.jsonl").read_text().splitlines()]
by_file = collections.defaultdict(list)
for r in rows: by_file[r["source_file"]].append(r)
for f, rs in by_file.items():
    arts = [r["article"] for r in rs if r["article"]]
    lens = [len(r["text"]) for r in rs]
    print(f"\n{f}: {len(rs)} chunks | articles {len(set(arts))} | "
          f"no-article {sum(1 for r in rs if not r['article'])} | "
          f"continuations {sum(1 for r in rs if r['part']>0)} | "
          f"len min/median/max {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    dup = [a for a,c in collections.Counter(arts).items() if c>1]
    print("  first articles:", sorted(set(arts), key=lambda a:(len(a),a))[:8], "| repeated:", dup[:6])