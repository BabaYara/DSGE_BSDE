from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "snippet",
    [
        "def sdfLog",
        "lemma sdf_log_simplify",
        "theorem sdf_drift_algebra",
        "lemma aggregate_consumption_drift",
        "lemma drift_residual_zero",
        "lemma face_order_cancellation",
    ],
)
def test_lucasnotes_contains_expected_snippets(snippet: str) -> None:
    src = Path("lean/LucasNotes.lean").read_text(encoding="utf-8")
    assert snippet in src, f"Expected {snippet!r} in LucasNotes.lean"
