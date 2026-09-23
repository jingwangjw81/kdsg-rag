import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.ingest import split_articles, window

SAMPLE = """Kantonales Datenschutzgesetz (KDSG)
1 Allgemeine Bestimmungen
Art. 1 Zweck
1 Dieses Gesetz bezweckt den Schutz.
Art. 2 Begriffe
1 In diesem Gesetz bedeuten:
a Personendaten: Angaben.
Art. 21a Registerpflicht
1 Die Behörden führen ein Register.
"""

def test_split_articles_finds_all_articles():
    parts = list(split_articles(SAMPLE))
    arts = [a for a, _ in parts]
    assert arts == [None, "1", "2", "21a"]
    assert parts[1][1].startswith("Art. 1")
    assert "Register" in parts[3][1]

def test_window_overlaps():
    text = "x" * 4000
    pieces = list(window(text, size=1800, overlap=200))
    assert len(pieces) == 3
    assert all(len(p) <= 1800 for p in pieces)
