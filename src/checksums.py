"""Compute SHA-256 for every file in sources.yaml and write it back."""
import hashlib, yaml, pathlib

RAW = pathlib.Path("data/raw")
meta_path = RAW / "sources.yaml"
meta = yaml.safe_load(meta_path.read_text())

for s in meta["sources"]:
    p = RAW / s["file"]
    if not p.exists():
        print(f"MISSING: {p}")
        continue
    s["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f"{s['file']}: {s['sha256'][:12]}…  {p.stat().st_size // 1024} KB")

meta_path.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))