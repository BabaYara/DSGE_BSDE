from pathlib import Path


def test_online_research_sections_present():
    p = Path("docs/online_research.md")
    txt = p.read_text()
    for header in ("Purpose", "Status", "Topics & Leads", "Key references", "Implementation notes", "Action Items"):
        assert header in txt, f"Missing section: {header}"


def test_replication_checklist_exists_and_has_sections():
    p = Path("docs/replication_checklist.md")
    txt = p.read_text()
    for header in ("Purpose", "Pre‑run", "Model wiring", "Notebook outputs", "CLI checks", "Tests", "Strict gating", "Provenance"):
        assert header in txt, f"Missing section: {header}"

