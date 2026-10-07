import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.rag import wants_history, build_filter, build_prompt, label

HIT = {"id": "1_KDSG_new__main__art21__0", "text": "Art. 21 Registerpflicht\n1 Die Behörden führen ein Register.",
       "distance": 0.3, "title": "Kantonales Datenschutzgesetz (KDSG)", "legal_ref": "BSG 152.04",
       "section": "", "article": "21", "role": "primary", "status": "in force",
       "valid_from": "2026-09-01", "valid_until": ""}

def test_history_trigger_words():
    assert wants_history("Was hat sich bei der Registerpflicht geändert?")
    assert wants_history("Was galt bisher für Datensammlungen?")
    assert not wants_history("Welche Behörden führen ein Register?")

def test_filter_excludes_historical_by_default():
    assert build_filter(False) == {"role": {"$ne": "historical"}}
    assert build_filter(True) is None

def test_label_and_prompt():
    assert label(HIT) == "Kantonales Datenschutzgesetz (KDSG), BSG 152.04, Art. 21, (in force)"
    p = build_prompt("Wer führt ein Register?", [HIT])
    assert "[1] Kantonales Datenschutzgesetz" in p
    assert "AUSSCHLIESSLICH" in p and "FRAGE: Wer führt ein Register?" in p