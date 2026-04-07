import pytest

from bt5151_credit_risk.skill_prompts import load_skill_prompt


def test_load_skill_prompt_reads_skill_file():
    prompt = load_skill_prompt("train-models")
    assert "train-models" in prompt


def test_load_skill_prompt_raises_for_missing_file():
    missing_name = "does-not-exist"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_skill_prompt(missing_name)

    assert str(exc_info.value) == f"Skill prompt not found: {missing_name}"


@pytest.mark.parametrize("skill_name", ["../README", "/tmp/skill", "nested/../../README"])
def test_load_skill_prompt_rejects_path_traversal(skill_name):
    with pytest.raises(ValueError) as exc_info:
        load_skill_prompt(skill_name)

    assert str(exc_info.value) == f"Invalid skill name: {skill_name}"
