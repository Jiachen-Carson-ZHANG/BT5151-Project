from pathlib import Path

import pytest

from bt5151_credit_risk.skill_prompts import load_skill_prompt


def test_load_skill_prompt_reads_skill_file():
    prompt = load_skill_prompt("train-models")
    assert "train-models" in prompt


def test_load_skill_prompt_raises_for_missing_file():
    missing_name = "does-not-exist"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_skill_prompt(missing_name)

    assert missing_name in str(exc_info.value)
    assert Path("skills") not in Path(str(exc_info.value)).parents
