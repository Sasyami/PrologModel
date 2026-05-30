from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - optional during offline tests
    requests = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.model_runtime import TrainingModelRuntime
from data.pipeline_constants import (
    ANNOTATED_REPOS_DIR,
    DEDUP_COSINE_THRESHOLD,
    DUPLICATE_LOG_PATH,
    INSTRUCTION_PROMPT_LINES,
    INSTRUCTION_PROMPT_VERSION,
    INSTRUCTION_README_CONTEXT_LINE,
    LLM_TIMEOUT_SEC,
    MAX_README_PROMPT_CHARS,
    MAX_TRAIN_SAMPLE_TOKENS,
    OLLAMA_GENERATE_URL,
    OLLAMA_MODEL,
    REPOS_DIR,
)
from data.pipeline_utils import (
    append_jsonl,
    build_training_messages,
    cosine_similarity,
    delete_sample_files,
    load_json,
    metadata_sort_key,
    normalize_instruction_text,
    save_json,
)


def build_instruction_prompt(code: str, readme_text: str = "") -> str:
    prompt = [*INSTRUCTION_PROMPT_LINES, "", "Код файла:", code.strip()]

    if readme_text:
        prompt.extend(
            [
                "",
                INSTRUCTION_README_CONTEXT_LINE,
                readme_text[:MAX_README_PROMPT_CHARS].strip(),
            ]
        )

    return "\n".join(prompt)


def generate_instruction(
    code: str,
    readme_text: str,
    model: str,
    url: str,
) -> str | None:
    if requests is None:
        raise RuntimeError("requests is required to call the Ollama HTTP API")

    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": build_instruction_prompt(code, readme_text),
            "stream": False,
            "temperature": 0.1,
        },
        timeout=LLM_TIMEOUT_SEC,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()

    raw_text = response.json().get("response", "")
    normalized = normalize_instruction_text(raw_text)
    return normalized or None


def _load_readme(meta: dict[str, Any], repos_root: Path) -> str:
    readme_rel_path = meta.get("readme_rel_path")
    if not readme_rel_path:
        return ""

    readme_path = repos_root / meta["repo_slug"] / readme_rel_path
    if not readme_path.exists():
        return ""
    return readme_path.read_text(encoding="utf-8", errors="ignore")


def _candidate_json_paths(annotated_root: Path) -> list[Path]:
    json_paths = sorted(annotated_root.rglob("*.json"))
    metas = [(path, load_json(path)) for path in json_paths]
    metas.sort(key=lambda item: metadata_sort_key(item[1]))
    return [path for path, _ in metas]


def _format_duration(total_seconds: float) -> str:
    seconds = max(0, int(total_seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _build_progress_prefix(index: int, total: int, started_at: float) -> str:
    percent = (index / total * 100.0) if total else 100.0
    elapsed = time.monotonic() - started_at
    if index > 0 and elapsed > 0:
        remaining = (elapsed / index) * max(total - index, 0)
        eta_text = _format_duration(remaining)
    else:
        eta_text = "--:--"
    return (
        f"[{index}/{total} {percent:5.1f}% | "
        f"elapsed {_format_duration(elapsed)} | eta {eta_text}]"
    )


def _can_skip_hydration(
    meta: dict[str, Any],
    runtime: TrainingModelRuntime,
    ollama_model: str,
    pl_path: Path,
    txt_path: Path,
    force: bool,
) -> bool:
    return (
        not force
        and pl_path.exists()
        and txt_path.exists()
        and bool(meta.get("instruction_sha256"))
        and meta.get("instruction_model") == ollama_model
        and meta.get("instruction_prompt_version") == INSTRUCTION_PROMPT_VERSION
        and meta.get("instruction_embedding_model") == runtime.model_id
        and meta.get("train_tokenizer_model_id") == runtime.model_id
        and meta.get("instruction_embedding") is not None
        and meta.get("train_sample_token_count") is not None
    )


def _hydrate_instruction(
    meta_path: Path,
    runtime: TrainingModelRuntime,
    ollama_model: str,
    repos_root: Path,
    annotated_root: Path,
    force: bool,
) -> tuple[dict[str, Any] | None, str, str | None]:
    meta = load_json(meta_path)
    pl_path = meta_path.with_suffix(".pl")
    txt_path = meta_path.with_suffix(".txt")

    if not pl_path.exists():
        return None, "missing_source", None

    if _can_skip_hydration(
        meta,
        runtime=runtime,
        ollama_model=ollama_model,
        pl_path=pl_path,
        txt_path=txt_path,
        force=force,
    ):
        return meta, "ready", None

    code = pl_path.read_text(encoding="utf-8").strip()
    instruction = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
    generated_now = False

    if force or not instruction:
        readme_text = _load_readme(meta, repos_root)
        try:
            generated = generate_instruction(
                code=code,
                readme_text=readme_text,
                model=ollama_model,
                url=OLLAMA_GENERATE_URL,
            )
        except Exception as exc:
            return None, "generation_failed", str(exc)

        if not generated:
            return None, "empty_instruction", None

        instruction = generated
        generated_now = True
        txt_path.write_text(instruction + "\n", encoding="utf-8")

    messages = build_training_messages(instruction, code)
    train_sample_token_count = runtime.count_chat_tokens(messages)
    if train_sample_token_count > MAX_TRAIN_SAMPLE_TOKENS:
        delete_sample_files(meta, repos_root=repos_root, annotated_root=annotated_root)
        return (
            None,
            "dropped_too_long",
            f"{train_sample_token_count} > {MAX_TRAIN_SAMPLE_TOKENS}",
        )

    if force or meta.get("instruction_embedding") is None:
        embedding = runtime.embed_text(instruction)
    else:
        embedding = meta["instruction_embedding"]

    meta["instruction_model"] = ollama_model
    meta["instruction_prompt_version"] = INSTRUCTION_PROMPT_VERSION
    meta["instruction_sha256"] = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    meta["instruction_embedding_model"] = runtime.model_id
    meta["instruction_embedding"] = embedding
    meta["train_tokenizer_model_id"] = runtime.model_id
    meta["train_sample_token_count"] = train_sample_token_count
    save_json(meta_path, meta)
    if generated_now:
        return meta, "generated", None
    return meta, "updated", None


def generate_file_descriptions(
    repos_root: Path = REPOS_DIR,
    annotated_root: Path = ANNOTATED_REPOS_DIR,
    ollama_model: str = OLLAMA_MODEL,
    force: bool = False,
) -> dict[str, int]:
    runtime = TrainingModelRuntime()
    kept: list[dict[str, Any]] = []
    candidate_paths = _candidate_json_paths(annotated_root)
    total_candidates = len(candidate_paths)
    started_at = time.monotonic()
    stats = {
        "total": total_candidates,
        "kept": 0,
        "duplicates": 0,
        "ready": 0,
        "generated": 0,
        "updated": 0,
        "empty_instruction": 0,
        "generation_failed": 0,
        "dropped_too_long": 0,
        "missing_source": 0,
    }

    print("=== Description generation ===")
    print(f"candidates: {total_candidates}")

    for index, meta_path in enumerate(candidate_paths, start=1):
        progress_prefix = _build_progress_prefix(index, total_candidates, started_at)
        meta, hydrate_status, detail = _hydrate_instruction(
            meta_path=meta_path,
            runtime=runtime,
            ollama_model=ollama_model,
            repos_root=repos_root,
            annotated_root=annotated_root,
            force=force,
        )
        if hydrate_status in stats:
            stats[hydrate_status] += 1

        if meta is None:
            source_label = meta_path.with_suffix(".pl").relative_to(annotated_root).as_posix()
            if hydrate_status == "empty_instruction":
                print(f"{progress_prefix} [skip] empty instruction {source_label}")
            elif hydrate_status == "missing_source":
                print(f"{progress_prefix} [skip] missing source {source_label}")
            elif hydrate_status == "generation_failed":
                print(f"{progress_prefix} [fail] instruction generation {source_label}: {detail}")
            elif hydrate_status == "dropped_too_long":
                print(f"{progress_prefix} [drop] sample too long {source_label}: {detail}")
            continue

        sample_label = f"{meta['repo_slug']}/{meta['source_rel_path']}"
        if hydrate_status == "ready":
            print(f"{progress_prefix} [skip] ready {sample_label}")
        elif hydrate_status == "generated":
            print(f"{progress_prefix} [desc] generated {sample_label}")
        elif hydrate_status == "updated":
            print(f"{progress_prefix} [desc] updated {sample_label}")

        duplicate_match: tuple[dict[str, Any], float] | None = None
        for kept_meta in kept:
            similarity = cosine_similarity(
                meta["instruction_embedding"],
                kept_meta["instruction_embedding"],
            )
            if similarity >= DEDUP_COSINE_THRESHOLD:
                duplicate_match = (kept_meta, similarity)
                break

        if duplicate_match is not None:
            canonical_meta, similarity = duplicate_match
            append_jsonl(
                DUPLICATE_LOG_PATH,
                {
                    "deleted_repo_slug": meta["repo_slug"],
                    "deleted_rel_path": meta["source_rel_path"],
                    "canonical_repo_slug": canonical_meta["repo_slug"],
                    "canonical_rel_path": canonical_meta["source_rel_path"],
                    "cosine_similarity": similarity,
                    "deleted_stripped_code_token_count": meta["stripped_code_token_count"],
                    "deleted_train_sample_token_count": meta["train_sample_token_count"],
                },
            )
            delete_sample_files(meta, repos_root=repos_root, annotated_root=annotated_root)
            stats["duplicates"] += 1
            print(
                f"{progress_prefix} [dup ] {meta['source_rel_path']} -> "
                f"{canonical_meta['repo_slug']}/{canonical_meta['source_rel_path']}"
            )
            continue

        kept.append(meta)
        stats["kept"] += 1
        print(f"{progress_prefix} [keep] {sample_label}")

    print("\n=== Annotation summary ===")
    print(f"candidates: {stats['total']}")
    print(f"kept: {stats['kept']}")
    print(f"ready_reused: {stats['ready']}")
    print(f"generated: {stats['generated']}")
    print(f"updated: {stats['updated']}")
    print(f"duplicates_removed: {stats['duplicates']}")
    print(f"dropped_too_long: {stats['dropped_too_long']}")
    print(f"empty_instruction: {stats['empty_instruction']}")
    print(f"generation_failed: {stats['generation_failed']}")
    print(f"missing_source: {stats['missing_source']}")
    print(f"elapsed: {_format_duration(time.monotonic() - started_at)}")
    return stats


def main(
    force: bool = False,
    ollama_model: str = OLLAMA_MODEL,
) -> dict[str, int]:
    return generate_file_descriptions(ollama_model=ollama_model, force=force)


if __name__ == "__main__":
    main()
