from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.model_runtime import TrainingModelRuntime
from data.pipeline_constants import (
    COMPILE_TIMEOUT_SEC,
    MAX_STRIPPED_CODE_TOKENS,
    MIN_COMPLEXITY_SCORE,
)
from data.pipeline_utils import (
    analyze_code_features,
    compute_complexity_score,
    compute_text_sha256,
    passes_complexity_threshold,
    strip_prolog_comments,
)


@dataclass
class PrologFileAnalysis:
    source_path: Path
    source_rel_path: str
    raw_code: str
    source_sha256: str
    stripped_code: str
    raw_bytes: int
    stripped_bytes: int
    stripped_code_token_count: int
    snapshot_tokenizer_model_id: str
    clause_count: int
    unique_predicate_count: int
    import_count: int
    has_recursion: bool
    control_marker_count: int
    complexity_score: int

    def to_metadata(self, repo_slug: str, readme_rel_path: str | None) -> dict[str, Any]:
        return {
            "repo_slug": repo_slug,
            "source_rel_path": self.source_rel_path.replace("\\", "/"),
            "readme_rel_path": readme_rel_path,
            "source_sha256": self.source_sha256,
            "train_tokenizer_model_id": None,
            "snapshot_tokenizer_model_id": self.snapshot_tokenizer_model_id,
            "snapshot_max_stripped_code_tokens": MAX_STRIPPED_CODE_TOKENS,
            "snapshot_min_complexity_score": MIN_COMPLEXITY_SCORE,
            "compile_ok": True,
            "raw_bytes": self.raw_bytes,
            "stripped_bytes": self.stripped_bytes,
            "clause_count": self.clause_count,
            "unique_predicate_count": self.unique_predicate_count,
            "import_count": self.import_count,
            "has_recursion": self.has_recursion,
            "control_marker_count": self.control_marker_count,
            "complexity_score": self.complexity_score,
            "stripped_code_token_count": self.stripped_code_token_count,
            "train_sample_token_count": None,
            "instruction_model": None,
            "instruction_sha256": None,
            "instruction_embedding_model": None,
            "instruction_embedding": None,
        }


def is_prolog_compilable(file_path: Path, timeout: int = COMPILE_TIMEOUT_SEC) -> bool:
    swipl = shutil.which("swipl")
    if not swipl:
        print("   [warn] SWI-Prolog not found in PATH; compile check skipped.")
        return True

    try:
        result = subprocess.run(
            [swipl, "-q", "--no-autoload", "-c", str(file_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"   [skip] compile timeout: {file_path.name}")
        return False
    except Exception as exc:
        print(f"   [skip] compile failed for {file_path.name}: {exc}")
        return False

    qlf_path = file_path.with_suffix(".qlf")
    if qlf_path.exists():
        qlf_path.unlink(missing_ok=True)

    return result.returncode == 0
def analyze_prolog_file(
    file_path: Path,
    repo_root: Path,
    runtime: TrainingModelRuntime,
    raw_code: str | None = None,
) -> PrologFileAnalysis | None:
    if raw_code is None:
        try:
            raw_code = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            print(f"   [skip] unable to read {file_path.name}: {exc}")
            return None

    if not is_prolog_compilable(file_path):
        return None

    stripped_code = strip_prolog_comments(raw_code)
    if not stripped_code:
        return None

    features = analyze_code_features(stripped_code)
    if features["clause_count"] == 0:
        return None

    complexity_score = compute_complexity_score(features)
    if not passes_complexity_threshold(features, MIN_COMPLEXITY_SCORE):
        print(
            f"   [skip] {file_path.name}: complexity score is too low "
            f"({complexity_score} < {MIN_COMPLEXITY_SCORE})"
        )
        return None

    token_count = runtime.count_tokens(stripped_code)
    if token_count > MAX_STRIPPED_CODE_TOKENS:
        print(
            f"   [skip] {file_path.name}: stripped code is too long "
            f"({token_count} > {MAX_STRIPPED_CODE_TOKENS} tokens)"
        )
        return None

    return PrologFileAnalysis(
        source_path=file_path,
        source_rel_path=str(file_path.relative_to(repo_root)),
        raw_code=raw_code,
        source_sha256=compute_text_sha256(raw_code),
        stripped_code=stripped_code,
        raw_bytes=len(raw_code.encode("utf-8", errors="ignore")),
        stripped_bytes=len(stripped_code.encode("utf-8", errors="ignore")),
        stripped_code_token_count=token_count,
        snapshot_tokenizer_model_id=runtime.model_id,
        clause_count=features["clause_count"],
        unique_predicate_count=features["unique_predicate_count"],
        import_count=features["import_count"],
        has_recursion=features["has_recursion"],
        control_marker_count=features["control_marker_count"],
        complexity_score=complexity_score,
    )
