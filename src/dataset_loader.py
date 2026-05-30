from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.model_runtime import TrainingModelRuntime
from data.pipeline_constants import ANNOTATED_REPOS_DIR, MAX_TRAIN_SAMPLE_TOKENS
from data.pipeline_utils import build_training_messages, load_json


@dataclass
class PrologSample:
    repo_slug: str
    source_rel_path: str
    instruction: str
    code: str
    train_sample_token_count: int
    messages: list[dict[str, str]]


def load_prolog_samples(
    data_dir: str | Path,
    max_train_sample_tokens: int = MAX_TRAIN_SAMPLE_TOKENS,
    recalc_token_counts: bool = False,
    runtime: TrainingModelRuntime | None = None,
) -> list[PrologSample]:
    samples: list[PrologSample] = []
    annotated_root = Path(data_dir) / ANNOTATED_REPOS_DIR.name
    if not annotated_root.exists():
        raise ValueError(f"Folder not found: {annotated_root}")

    for meta_path in sorted(annotated_root.rglob("*.json")):
        meta = load_json(meta_path)
        pl_path = meta_path.with_suffix(".pl")
        txt_path = meta_path.with_suffix(".txt")
        if not pl_path.exists() or not txt_path.exists():
            continue

        instruction = txt_path.read_text(encoding="utf-8").strip()
        code = pl_path.read_text(encoding="utf-8").strip()
        messages = build_training_messages(instruction, code)

        train_sample_token_count = meta.get("train_sample_token_count")
        if recalc_token_counts or train_sample_token_count is None:
            runtime = runtime or TrainingModelRuntime()
            train_sample_token_count = runtime.count_chat_tokens(messages)

        if int(train_sample_token_count) > max_train_sample_tokens:
            continue

        samples.append(
            PrologSample(
                repo_slug=meta["repo_slug"],
                source_rel_path=meta["source_rel_path"],
                instruction=instruction,
                code=code,
                train_sample_token_count=int(train_sample_token_count),
                messages=messages,
            )
        )

    return samples


def load_prolog_data(
    data_dir: str | Path,
    max_train_sample_tokens: int = MAX_TRAIN_SAMPLE_TOKENS,
    recalc_token_counts: bool = False,
    runtime: TrainingModelRuntime | None = None,
) -> list[list[dict[str, str]]]:
    samples = load_prolog_samples(
        data_dir=data_dir,
        max_train_sample_tokens=max_train_sample_tokens,
        recalc_token_counts=recalc_token_counts,
        runtime=runtime,
    )
    return [sample.messages for sample in samples]


def split_samples_by_repo(
    samples: list[PrologSample],
    test_size: float = 0.1,
    seed: int = 42,
) -> dict[str, list[PrologSample]]:
    if not samples:
        return {"train": [], "test": []}

    grouped: dict[str, list[PrologSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.repo_slug, []).append(sample)

    repo_slugs = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(repo_slugs)

    test_repo_count = max(1, round(len(repo_slugs) * test_size))
    test_repos = set(repo_slugs[:test_repo_count])

    train: list[PrologSample] = []
    test: list[PrologSample] = []
    for repo_slug, repo_samples in grouped.items():
        if repo_slug in test_repos:
            test.extend(repo_samples)
        else:
            train.extend(repo_samples)

    return {"train": train, "test": test}
