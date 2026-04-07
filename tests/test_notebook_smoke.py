from pathlib import Path


def test_notebook_exists():
    assert Path("bt5151_credit_risk_pipeline.ipynb").exists()
