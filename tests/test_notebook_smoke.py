from pathlib import Path


def test_notebook_exists():
    assert Path("bt5151_credit_risk_pipeline.ipynb").exists()


def test_notebook_uses_compiled_graph():
    notebook_text = Path("bt5151_credit_risk_pipeline.ipynb").read_text()
    assert "compile_graph" in notebook_text
