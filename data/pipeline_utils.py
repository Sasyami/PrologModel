from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

from data.pipeline_constants import (
    IGNORED_DIR_NAMES,
    README_CANDIDATES,
    TRAINING_SYSTEM_PROMPT,
)


CLAUSE_HEAD_RE = re.compile(
    r"^\s*(?!:-)([a-z_][a-zA-Z0-9_]*)\s*(?:\((.*?)\))?\s*(?::-|\.)",
    re.MULTILINE | re.DOTALL,
)
IMPORT_RE = re.compile(r"\b(?:use_module|ensure_loaded)\b")
CONTROL_MARKERS = (
    r"\bfindall\(",
    r"\bmaplist\(",
    r"\bmember\(",
    r"\bappend\(",
    r"->",
    r";",
    r"\bis\b",
    r"\bcatch\(",
    r"\brepeat\b",
    r"!\s*[,\.\)]",
    r"\\\+",
)


def repo_slug(repo_full_name: str) -> str:
    return repo_full_name.replace("/", "_")


def compute_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_prolog_comments(code: str) -> str:
    lines = code.splitlines()
    cleaned_lines: list[str] = []
    in_block_comment = False

    for line in lines:
        cleaned_line = ""
        index = 0

        while index < len(line):
            if in_block_comment:
                if line[index : index + 2] == "*/":
                    in_block_comment = False
                    index += 2
                    continue
                index += 1
                continue

            if line[index : index + 2] == "/*":
                in_block_comment = True
                index += 2
                continue

            if line[index] == "%":
                break

            if line[index] in "\"'":
                quote = line[index]
                cleaned_line += quote
                index += 1
                while index < len(line):
                    if line[index] == "\\" and index + 1 < len(line):
                        cleaned_line += line[index : index + 2]
                        index += 2
                    elif line[index] == quote:
                        cleaned_line += line[index]
                        index += 1
                        break
                    else:
                        cleaned_line += line[index]
                        index += 1
                continue

            cleaned_line += line[index]
            index += 1

        cleaned_lines.append(cleaned_line.rstrip())

    result = "\n".join(cleaned_lines)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def iter_top_level_terms(code: str) -> list[str]:
    terms: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    escape_next = False

    for char in code:
        current.append(char)

        if escape_next:
            escape_next = False
            continue

        if quote is not None:
            if char == "\\":
                escape_next = True
            elif char == quote:
                quote = None
            continue

        if char in "\"'":
            quote = char
            continue

        if char in "([{":
            depth += 1
            continue

        if char in ")]}":
            depth = max(depth - 1, 0)
            continue

        if char == "." and depth == 0:
            term = "".join(current).strip()
            if term:
                terms.append(term)
            current = []

    tail = "".join(current).strip()
    if tail:
        terms.append(tail)
    return terms


def analyze_code_features(stripped_code: str) -> dict[str, Any]:
    terms = iter_top_level_terms(stripped_code)
    predicates: list[str] = []
    has_recursion = False

    for term in terms:
        if term.startswith(":-"):
            continue
        match = CLAUSE_HEAD_RE.match(term)
        if not match:
            continue
        predicate_name = match.group(1)
        predicates.append(predicate_name)

        if ":-" in term:
            _, body = term.split(":-", 1)
            recursion_re = rf"\b{re.escape(predicate_name)}\s*(?:\(|\.)"
            if re.search(recursion_re, body):
                has_recursion = True

    control_marker_count = sum(
        1 for marker in CONTROL_MARKERS if re.search(marker, stripped_code)
    )
    import_count = len(IMPORT_RE.findall(stripped_code))
    unique_predicates = len(set(predicates))

    return {
        "clause_count": len(predicates),
        "unique_predicate_count": unique_predicates,
        "import_count": import_count,
        "has_recursion": has_recursion,
        "control_marker_count": control_marker_count,
    }


def compute_complexity_score(features: dict[str, Any]) -> int:
    return (
        int(features.get("clause_count", 0))
        + int(features.get("unique_predicate_count", 0))
        + int(features.get("control_marker_count", 0))
        + int(features.get("import_count", 0))
        + (2 if features.get("has_recursion") else 0)
    )


def passes_complexity_threshold(features: dict[str, Any], threshold: int) -> bool:
    return compute_complexity_score(features) >= threshold


def has_user_clauses(stripped_code: str) -> bool:
    return analyze_code_features(stripped_code)["clause_count"] > 0


def iter_prolog_files(root: Path) -> list[Path]:
    prolog_files: list[Path] = []
    for current_root, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIR_NAMES)
        for filename in sorted(files):
            if Path(filename).suffix.lower() == ".pl":
                prolog_files.append(Path(current_root) / filename)
    return sorted(prolog_files)


def find_readme(root: Path) -> Path | None:
    for candidate in README_CANDIDATES:
        readme_path = root / candidate
        if readme_path.exists() and readme_path.is_file():
            return readme_path
    return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_instruction_text(text: str) -> str:
    normalized = text.replace("```prolog", "").replace("```", "").replace("`", "")
    normalized = normalized.replace("\r\n", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized.strip()


def build_training_messages(
    instruction: str,
    code: str,
    system_prompt: str = TRAINING_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": f"\n{code.strip()}\n"},
    ]


def metadata_sort_key(meta: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(meta.get("complexity_score", 0)),
        -int(bool(meta.get("has_recursion"))),
        -int(meta.get("control_marker_count", 0)),
        -int(meta.get("unique_predicate_count", 0)),
        -int(meta.get("clause_count", 0)),
        int(meta.get("stripped_code_token_count", 0)),
        str(meta.get("repo_slug", "")),
        str(meta.get("source_rel_path", "")),
    )


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mirrored_paths(
    repos_root: Path,
    annotated_root: Path,
    repo_slug_value: str,
    source_rel_path: str,
) -> dict[str, Path]:
    relative_path = Path(source_rel_path)
    source_path = repos_root / repo_slug_value / relative_path
    annotated_pl = annotated_root / repo_slug_value / relative_path
    return {
        "source": source_path,
        "annotated_pl": annotated_pl,
        "annotated_txt": annotated_pl.with_suffix(".txt"),
        "annotated_json": annotated_pl.with_suffix(".json"),
        "repo_root": repos_root / repo_slug_value,
        "annotated_root": annotated_root / repo_slug_value,
    }


def prune_empty_parent_dirs(path: Path, stop_at: Path) -> None:
    current = path
    while current.exists() and current != stop_at:
        try:
            next(current.iterdir())
        except StopIteration:
            current.rmdir()
            current = current.parent
            continue
        break


def remove_tree_if_no_prolog(repo_root: Path) -> None:
    if not repo_root.exists():
        return
    if any(repo_root.rglob("*.pl")):
        return
    shutil.rmtree(repo_root)


def delete_sample_files(
    meta: dict[str, Any],
    repos_root: Path,
    annotated_root: Path,
) -> None:
    paths = mirrored_paths(
        repos_root,
        annotated_root,
        meta["repo_slug"],
        meta["source_rel_path"],
    )
    for key in ("source", "annotated_pl", "annotated_txt", "annotated_json"):
        path = paths[key]
        if path.exists():
            path.unlink()

    if paths["annotated_root"].exists():
        prune_empty_parent_dirs(paths["annotated_pl"].parent, paths["annotated_root"])
    if paths["repo_root"].exists():
        prune_empty_parent_dirs(paths["source"].parent, paths["repo_root"])

    remove_tree_if_no_prolog(paths["annotated_root"])
    remove_tree_if_no_prolog(paths["repo_root"])
