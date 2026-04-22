from pathlib import Path


def test_citation_bib_exists_and_mentions_arxiv() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    content = (repo_root / "CITATION.bib").read_text()
    assert "@article" in content
    assert "2012.14853" in content
    assert "Mirza" in content
