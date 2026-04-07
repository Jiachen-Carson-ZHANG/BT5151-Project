from pathlib import Path


def load_skill_prompt(skill_name: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    skill_path = repo_root / "skills" / f"{skill_name}.md"

    if not skill_path.exists():
        raise FileNotFoundError(f"Skill prompt not found: {skill_path}")

    return skill_path.read_text(encoding="utf-8")
