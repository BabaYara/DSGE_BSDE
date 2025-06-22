from pathlib import Path


def test_agent_task_template_fields() -> None:
    template = Path(".github/PULL_REQUEST_TEMPLATE/agent_task.yaml")
    assert template.exists(), "agent_task.yaml missing"
    text = template.read_text()
    required = [
        "milestone",
        "files_touched",
        "tests_added",
        "notebooks_updated",
        "ci_passed",
    ]
    for key in required:
        assert f"{key}:" in text, f"{key} missing from agent_task.yaml"
