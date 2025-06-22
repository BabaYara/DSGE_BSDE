from pathlib import Path


def test_docs_reference_changelog() -> None:
    index = Path("docs/index.md")
    assert index.exists(), "docs/index.md missing"
    text = index.read_text()
    assert "../CHANGELOG.md" in text, "Changelog link missing from docs/index.md"
