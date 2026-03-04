import shutil
import re
from pathlib import Path


# -----------------------------
# Frontmatter Utilities
# -----------------------------


def normalize_filename(name: str) -> str:
    """
    Replace ',' and space with '_' and lowercase everything.
    """
    return name.replace(",", "_").replace(" ", "_").lower()


def parse_frontmatter(lines):
    if not lines or lines[0].strip() != "---":
        return None, None

    frontmatter = []
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return frontmatter, i + 1
        frontmatter.append(line.rstrip("\n"))

    return None, None


def has_tag(frontmatter, target_tag):
    in_tags = False
    for line in frontmatter:
        stripped = line.strip()

        if stripped.startswith("tags:"):
            in_tags = True
            continue

        if in_tags:
            if stripped.startswith("-"):
                tag = stripped.lstrip("-").strip()
                if tag == target_tag:
                    return True
            else:
                if stripped and not stripped.startswith("-"):
                    in_tags = False
    return False


def has_from_obsidian_flag(frontmatter):
    return any(line.strip().lower() == "from_obsidian: true" for line in frontmatter)


def add_from_obsidian_flag(frontmatter):
    if has_from_obsidian_flag(frontmatter):
        return frontmatter
    return frontmatter + ["from_obsidian: true"]


# -----------------------------
# Delete generated files
# -----------------------------


def delete_existing_flagged_files(target_dir: Path):
    for md_file in target_dir.rglob("*.md"):
        try:
            lines = md_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        frontmatter, _ = parse_frontmatter(lines)
        if frontmatter and has_from_obsidian_flag(frontmatter):
            md_file.unlink()
            print(f"Deleted: {md_file.relative_to(target_dir)}")


# -----------------------------
# Wiki link conversion
# -----------------------------


def build_note_index(source_dir: Path, tag: str):
    index = {}
    duplicates = {}

    for md_file in source_dir.rglob("*.md"):
        try:
            lines = md_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        frontmatter, _ = parse_frontmatter(lines)
        if not frontmatter or not has_tag(frontmatter, tag):
            continue

        normalized_name = normalize_filename(md_file.stem)
        normalized_relative_path = md_file.relative_to(source_dir).parent / (
            normalized_name + ".md"
        )

        if normalized_name in index:
            duplicates.setdefault(normalized_name, []).append(md_file)
        else:
            index[normalized_name] = normalized_relative_path

    if duplicates:
        print("WARNING: Duplicate note names after normalization!")

    return index


def convert_wiki_links(text, current_file: Path, note_index: dict, source_dir: Path):
    WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")

    current_relative = current_file.relative_to(source_dir).parent / (
        normalize_filename(current_file.stem) + ".md"
    )
    current_dir = current_relative.parent

    def replace(match):
        inner = match.group(1)
        if "|" in inner:
            target_name, alias = inner.split("|", 1)
            alias = alias.strip()
        else:
            target_name = inner
            alias = inner
        target_name = target_name.strip()

        if "#" in target_name:
            target_name = target_name.split("#", 1)[0]

        normalized_target = normalize_filename(target_name)

        if normalized_target not in note_index:
            return match.group(0)

        target_relative = note_index[normalized_target]

        rel_path = Path(target_relative).relative_to(current_dir)
        return f"[{alias}]({rel_path.as_posix()})"

    return WIKI_LINK_PATTERN.sub(replace, text)


# -----------------------------
# Main copy logic
# -----------------------------


def copy_matching_markdown_files(source_dir: Path, target_dir: Path, tag: str):
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()

    delete_existing_flagged_files(target_dir)

    note_index = build_note_index(source_dir, tag)

    for md_file in source_dir.rglob("*.md"):
        lines = md_file.read_text(encoding="utf-8").splitlines()
        frontmatter, content_start = parse_frontmatter(lines)

        if not frontmatter:
            continue

        if not has_tag(frontmatter, tag):
            continue

        updated_frontmatter = add_from_obsidian_flag(frontmatter)

        body = "\n".join(lines[content_start:])
        body = convert_wiki_links(body, md_file, note_index, source_dir)

        new_content = ["---"] + updated_frontmatter + ["---"] + body.splitlines()

        relative_path = md_file.relative_to(source_dir)
        normalized_name = normalize_filename(md_file.stem) + ".md"
        destination = target_dir / relative_path.parent / normalized_name
        destination.parent.mkdir(parents=True, exist_ok=True)

        destination.write_text("\n".join(new_content) + "\n", encoding="utf-8")
        print(f"Copied: {relative_path}")


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export Obsidian markdown with tag filtering and link conversion"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("tag", type=str)

    args = parser.parse_args()

    print("\nNOTE:")
    print("Does not yet preserve #heading anchor in converted link.\n")

    copy_matching_markdown_files(args.source, args.target, args.tag)
