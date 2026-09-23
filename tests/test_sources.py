import yaml, pathlib, hashlib

def test_sources_present_and_unchanged():
    raw = pathlib.Path("data/raw")
    meta = yaml.safe_load((raw / "sources.yaml").read_text())
    for s in meta["sources"]:
        p = raw / s["file"]
        assert p.exists(), s["file"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == s["sha256"], s["file"]