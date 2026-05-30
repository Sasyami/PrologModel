from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.model_runtime import CleanupCodeEmbeddingRuntime, TrainingModelRuntime
from data.pipeline_constants import (
    ANNOTATED_REPOS_DIR,
    CODE_CLEANUP_COSINE_THRESHOLD,
    MAX_STRIPPED_CODE_TOKENS,
    REPOS_DIR,
)
from data.pipeline_utils import (
    compute_text_sha256,
    delete_sample_files,
    load_json,
    metadata_sort_key,
)


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


def _find_semantic_duplicate(
    embedding: Any,
    kept_matrix: Any | None,
    pending_vectors: list[Any],
    torch: Any,
) -> tuple[int | None, float]:
    best_index: int | None = None
    best_similarity = -1.0
    matrix_size = 0

    if kept_matrix is not None and kept_matrix.numel() > 0:
        similarities = torch.mv(kept_matrix, embedding)
        similarity, index = similarities.max(dim=0)
        best_similarity = float(similarity.item())
        best_index = int(index.item())
        matrix_size = int(kept_matrix.shape[0])

    if pending_vectors:
        pending_matrix = torch.stack(pending_vectors)
        similarities = torch.mv(pending_matrix, embedding)
        similarity, index = similarities.max(dim=0)
        pending_similarity = float(similarity.item())
        if pending_similarity > best_similarity:
            best_similarity = pending_similarity
            best_index = matrix_size + int(index.item())

    return best_index, best_similarity


def _flush_pending_vectors(
    kept_matrix: Any | None,
    pending_vectors: list[Any],
    torch: Any,
) -> Any | None:
    if not pending_vectors:
        return kept_matrix

    chunk = torch.stack(pending_vectors)
    pending_vectors.clear()
    if kept_matrix is None:
        return chunk
    return torch.cat([kept_matrix, chunk], dim=0)


def cleanup_similar_code(
    repos_root: Path = REPOS_DIR,
    annotated_root: Path = ANNOTATED_REPOS_DIR,
) -> dict[str, int]:
    training_runtime = TrainingModelRuntime()
    cleanup_runtime = CleanupCodeEmbeddingRuntime()
    torch = cleanup_runtime.torch

    candidate_paths = _candidate_json_paths(annotated_root)
    total_candidates = len(candidate_paths)
    started_at = time.monotonic()
    kept_metas: list[dict[str, Any]] = []
    kept_matrix: Any | None = None
    pending_vectors: list[Any] = []
    seen_hashes: dict[str, dict[str, Any]] = {}
    flush_batch_size = 64

    stats = {
        "total": total_candidates,
        "kept": 0,
        "exact_duplicates": 0,
        "semantic_duplicates": 0,
        "dropped_too_long": 0,
        "missing_source": 0,
        "errors": 0,
    }

    print("=== Code cleanup ===")
    print(f"candidates: {total_candidates}")
    print(f"train_tokenizer_model: {training_runtime.model_id}")
    print(f"cleanup_embedding_model: {cleanup_runtime.model_id}")
    print(f"cleanup_cosine_threshold: {CODE_CLEANUP_COSINE_THRESHOLD}")

    for index, meta_path in enumerate(candidate_paths, start=1):
        progress_prefix = _build_progress_prefix(index, total_candidates, started_at)
        source_label = meta_path.with_suffix(".pl").relative_to(annotated_root).as_posix()

        try:
            meta = load_json(meta_path)
            pl_path = meta_path.with_suffix(".pl")
            if not pl_path.exists():
                stats["missing_source"] += 1
                print(f"{progress_prefix} [skip] missing source {source_label}")
                continue

            code = pl_path.read_text(encoding="utf-8", errors="ignore").strip()
            code_token_count = training_runtime.count_tokens(code)
            if code_token_count > MAX_STRIPPED_CODE_TOKENS:
                delete_sample_files(meta, repos_root=repos_root, annotated_root=annotated_root)
                stats["dropped_too_long"] += 1
                print(
                    f"{progress_prefix} [drop] too long {meta['repo_slug']}/{meta['source_rel_path']}: "
                    f"{code_token_count} > {MAX_STRIPPED_CODE_TOKENS}"
                )
                continue

            code_hash = compute_text_sha256(code)
            exact_match = seen_hashes.get(code_hash)
            if exact_match is not None:
                delete_sample_files(meta, repos_root=repos_root, annotated_root=annotated_root)
                stats["exact_duplicates"] += 1
                print(
                    f"{progress_prefix} [dup ] exact {meta['repo_slug']}/{meta['source_rel_path']} -> "
                    f"{exact_match['repo_slug']}/{exact_match['source_rel_path']}"
                )
                continue

            embedding = cleanup_runtime.embed_text_tensor(code)
            match_index, similarity = _find_semantic_duplicate(
                embedding=embedding,
                kept_matrix=kept_matrix,
                pending_vectors=pending_vectors,
                torch=torch,
            )
            if match_index is not None and similarity >= CODE_CLEANUP_COSINE_THRESHOLD:
                canonical_meta = kept_metas[match_index]
                delete_sample_files(meta, repos_root=repos_root, annotated_root=annotated_root)
                stats["semantic_duplicates"] += 1
                print(
                    f"{progress_prefix} [dup ] semantic {meta['repo_slug']}/{meta['source_rel_path']} -> "
                    f"{canonical_meta['repo_slug']}/{canonical_meta['source_rel_path']} "
                    f"(cos={similarity:.4f})"
                )
                continue

            kept_metas.append(meta)
            pending_vectors.append(embedding)
            seen_hashes[code_hash] = meta
            if len(pending_vectors) >= flush_batch_size:
                kept_matrix = _flush_pending_vectors(
                    kept_matrix=kept_matrix,
                    pending_vectors=pending_vectors,
                    torch=torch,
                )

            stats["kept"] += 1
            print(f"{progress_prefix} [keep] {meta['repo_slug']}/{meta['source_rel_path']}")
        except Exception as exc:
            stats["errors"] += 1
            print(f"{progress_prefix} [fail] {source_label}: {exc}")

    _flush_pending_vectors(
        kept_matrix=kept_matrix,
        pending_vectors=pending_vectors,
        torch=torch,
    )

    print("\n=== Cleanup summary ===")
    print(f"candidates: {stats['total']}")
    print(f"kept: {stats['kept']}")
    print(f"exact_duplicates_removed: {stats['exact_duplicates']}")
    print(f"semantic_duplicates_removed: {stats['semantic_duplicates']}")
    print(f"dropped_too_long: {stats['dropped_too_long']}")
    print(f"missing_source: {stats['missing_source']}")
    print(f"errors: {stats['errors']}")
    print(f"elapsed: {_format_duration(time.monotonic() - started_at)}")
    return stats


def main() -> dict[str, int]:
    return cleanup_similar_code()


if __name__ == "__main__":
    main()
