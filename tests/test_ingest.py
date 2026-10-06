import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.ingest import split_articles, window, cut_history

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

ANNEXED = """Weisung
Art. 1 Zweck
Diese Weisung regelt.
Anhang 5 zu ICSGW: Umgang mit Authentisierungsmerkmalen
Art. 1 Dieser Anhang regelt die Handhabung.
Art. 2 In diesem Anhang bedeuten:
Anhang 6 ICSGW Kryptographische Verfahren
1 Dieser Anhang regelt die Verwendung.
"""

AMENDED = """Art. 6 Protokollierung
1 Behörden protokollieren das Speichern und Löschen von Personendaten.
Der Erlass 861.112 Verordnung über die Datenbearbeitung vom 20.05.2020 wird wie folgt geändert:
Art. 6 Abs. 3 (geändert)
3 Die kantonale Steuerverwaltung protokolliert die Datenzugriffe.
"""

def test_split_articles_finds_all_articles():
    parts = list(split_articles(SAMPLE))
    assert [a for _, a, _ in parts] == [None, "1", "2", "21a"]
    assert parts[1][2].startswith("Art. 1")
    assert "Register" in parts[3][2]
    assert all(sec is None for sec, _, _ in parts)

def test_sections_make_article_numbers_unique():
    parts = list(split_articles(ANNEXED))
    keys = [(sec, a) for sec, a, _ in parts]
    assert (None, "1") in keys and ("Anhang 5", "1") in keys and ("Anhang 5", "2") in keys
    assert ("Anhang 6", None) in keys
    assert not any("Anhang 5 zu ICSGW" in body for _, _, body in parts)   # marker lines removed

def test_amendment_block_becomes_section():
    parts = list(split_articles(AMENDED))
    keys = [(sec, a) for sec, a, _ in parts]
    assert (None, "6") in keys and ("Änderung 861.112", "6") in keys
    assert any(body.startswith("Der Erlass 861.112") for sec, _, body in parts if sec == "Änderung 861.112")

def test_cut_history():
    assert cut_history("Art. 1 Text\nÄnderungstabelle\nArt. 1 geändert") == "Art. 1 Text\n"

def test_window_overlaps():
    pieces = list(window("x" * 4000, size=1800, overlap=200))
    assert len(pieces) == 3 and all(len(p) <= 1800 for p in pieces)

