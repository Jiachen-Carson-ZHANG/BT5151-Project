from pathlib import Path


def load_skill_prompt(skill_name: str) -> str:
    skill_name_path = Path(skill_name)
    if skill_name_path.is_absolute() or len(skill_name_path.parts) != 1:
        raise ValueError(f"Invalid skill name: {skill_name}")

    repo_root = Path(__file__).resolve().parents[2]
    skills_dir = (repo_root / "skills").resolve()
    skill_path = (skills_dir / f"{skill_name}.md").resolve()

    try:
        skill_path.relative_to(skills_dir)
    except ValueError as exc:
        raise ValueError(f"Invalid skill name: {skill_name}") from exc

    if not skill_path.exists():
        raise FileNotFoundError(f"Skill prompt not found: {skill_name}")

    return skill_path.read_text(encoding="utf-8")
