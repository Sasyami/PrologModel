from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.github_prolog_repos import get_prolog_repos
from data.github_utils import download_github_repo
from data.model_runtime import TrainingModelRuntime
from data.pipeline_constants import (
    ANNOTATED_REPOS_DIR,
    DOWNLOAD_DELAY_SEC,
    GITHUB_MAX_REPOS,
    GITHUB_MAX_STARS,
    GITHUB_MIN_STARS,
    MAX_STRIPPED_CODE_TOKENS,
    MIN_COMPLEXITY_SCORE,
    REPOS_DIR,
)
from data.pipeline_utils import (
    compute_text_sha256,
    delete_sample_files,
    find_readme,
    iter_prolog_files,
    load_json,
    remove_tree_if_no_prolog,
    repo_slug,
    save_json,
)
from data.prolog_analyze import PrologFileAnalysis, analyze_prolog_file


def _copy_readme(repo_root: Path, repo_target_root: Path) -> str | None:
    readme_path = find_readme(repo_root)
    if readme_path is None:
        return None

    destination = repo_target_root / readme_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(readme_path, destination)
    return readme_path.name


def _should_skip_repo_download(repo_name: str, repos_root: Path) -> bool:
    return (repos_root / repo_slug(repo_name)).exists()


def _write_analysis_snapshot(
    analysis: PrologFileAnalysis,
    repo_slug_value: str,
    readme_rel_path: str | None,
    repos_root: Path,
    annotated_root: Path,
) -> None:
    source_destination = repos_root / repo_slug_value / analysis.source_rel_path
    source_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(analysis.source_path, source_destination)

    annotated_destination = annotated_root / repo_slug_value / analysis.source_rel_path
    annotated_destination.parent.mkdir(parents=True, exist_ok=True)
    annotated_destination.write_text(analysis.stripped_code + "\n", encoding="utf-8")

    metadata = analysis.to_metadata(repo_slug_value, readme_rel_path)
    save_json(annotated_destination.with_suffix(".json"), metadata)


def _is_snapshot_current(
    meta: dict[str, Any],
    raw_code: str,
    runtime: TrainingModelRuntime,
) -> bool:
    return (
        meta.get("source_sha256") == compute_text_sha256(raw_code)
        and meta.get("snapshot_tokenizer_model_id") == runtime.model_id
        and meta.get("snapshot_max_stripped_code_tokens") == MAX_STRIPPED_CODE_TOKENS
        and meta.get("snapshot_min_complexity_score") == MIN_COMPLEXITY_SCORE
    )


def _remove_stale_snapshots(
    repo_slug_value: str,
    kept_rel_paths: set[str],
    repos_root: Path,
    annotated_root: Path,
) -> None:
    repo_annotated_root = annotated_root / repo_slug_value
    if repo_annotated_root.exists():
        for meta_path in sorted(repo_annotated_root.rglob("*.json")):
            meta = load_json(meta_path)
            source_rel_path = str(meta.get("source_rel_path", "")).replace("\\", "/")
            if source_rel_path in kept_rel_paths:
                continue
            delete_sample_files(meta, repos_root=repos_root, annotated_root=annotated_root)

    remove_tree_if_no_prolog(repos_root / repo_slug_value)
    remove_tree_if_no_prolog(annotated_root / repo_slug_value)


def build_repo_snapshot(
    repo_path: Path,
    repo_name: str,
    runtime: TrainingModelRuntime,
    repos_root: Path = REPOS_DIR,
    annotated_root: Path = ANNOTATED_REPOS_DIR,
) -> list[dict[str, Any]]:
    slug = repo_slug(repo_name)
    repo_source_root = repos_root / slug
    repo_annotated_root = annotated_root / slug

    repo_source_root.mkdir(parents=True, exist_ok=True)
    repo_annotated_root.mkdir(parents=True, exist_ok=True)
    readme_rel_path = _copy_readme(repo_path, repo_source_root)

    selected: list[dict[str, Any]] = []
    kept_rel_paths: set[str] = set()
    for file_path in iter_prolog_files(repo_path):
        source_rel_path = str(file_path.relative_to(repo_path)).replace("\\", "/")
        source_destination = repos_root / slug / source_rel_path
        annotated_destination = annotated_root / slug / source_rel_path
        meta_path = annotated_destination.with_suffix(".json")

        try:
            raw_code = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            print(f"   [skip] unable to read {file_path.name}: {exc}")
            continue

        if (
            source_destination.exists()
            and annotated_destination.exists()
            and meta_path.exists()
        ):
            meta = load_json(meta_path)
            if _is_snapshot_current(meta, raw_code=raw_code, runtime=runtime):
                kept_rel_paths.add(source_rel_path)
                selected.append(meta)
                print(f"   [skip] already processed {source_rel_path}")
                continue

        analysis = analyze_prolog_file(file_path, repo_path, runtime, raw_code=raw_code)
        if analysis is None:
            continue
        _write_analysis_snapshot(
            analysis,
            slug,
            readme_rel_path,
            repos_root,
            annotated_root,
        )
        kept_rel_paths.add(source_rel_path)
        selected.append(analysis.to_metadata(slug, readme_rel_path))
        print(f"   [keep] {analysis.source_rel_path}")

    _remove_stale_snapshots(
        slug,
        kept_rel_paths=kept_rel_paths,
        repos_root=repos_root,
        annotated_root=annotated_root,
    )

    if not selected:
        if repo_source_root.exists():
            shutil.rmtree(repo_source_root)
        if repo_annotated_root.exists():
            shutil.rmtree(repo_annotated_root)

    return selected


def build_repo_snapshots(
    repos: list[dict[str, Any]],
    repos_root: Path = REPOS_DIR,
    annotated_root: Path = ANNOTATED_REPOS_DIR,
    delay_between: float = DOWNLOAD_DELAY_SEC,
) -> list[dict[str, Any]]:
    repos_root.mkdir(parents=True, exist_ok=True)
    annotated_root.mkdir(parents=True, exist_ok=True)

    runtime = TrainingModelRuntime()
    reports: list[dict[str, Any]] = []

    for index, repo in enumerate(repos, start=1):
        repo_name = repo["full_name"]
        repo_slug_value = repo_slug(repo_name)
        print(f"\n[{index}/{len(repos)}] {repo_name}")
        print("=" * 60)

        if _should_skip_repo_download(repo_name, repos_root):
            selected_files = sum(1 for _ in (annotated_root / repo_slug_value).rglob("*.json"))
            reports.append(
                {
                    "repo_slug": repo_slug_value,
                    "repo_name": repo_name,
                    "selected_files": selected_files,
                    "skipped_existing_repo": True,
                }
            )
            print(f"[skip] repo already exists: {repo_slug_value}")
            continue

        with tempfile.TemporaryDirectory(prefix="tgsum_repo_") as temp_dir:
            downloaded_path = download_github_repo(repo_name, output_dir=temp_dir)
            if downloaded_path is None:
                print(f"[skip] unable to download {repo_name}")
                continue

            selected = build_repo_snapshot(
                Path(downloaded_path),
                repo_name,
                runtime,
                repos_root=repos_root,
                annotated_root=annotated_root,
            )

        report = {
            "repo_slug": repo_slug_value,
            "repo_name": repo_name,
            "selected_files": len(selected),
        }
        reports.append(report)
        print(f"[done] selected files: {len(selected)}")

        if index < len(repos):
            time.sleep(delay_between)

    print("\n=== Repo snapshot summary ===")
    print(f"processed: {len(reports)}")
    print(f"selected_files: {sum(item['selected_files'] for item in reports)}")
    return reports


def main(
    repo_names: list[str] | None = None,
    min_stars: int = GITHUB_MIN_STARS,
    max_stars: int | None = GITHUB_MAX_STARS or None,
    max_repos: int = GITHUB_MAX_REPOS,
    max_pages: int = 100,
    per_page: int = 30,
    delay_between: float = DOWNLOAD_DELAY_SEC,
) -> list[dict[str, Any]]:
    repos = (
        [{"full_name": repo_name} for repo_name in repo_names]
        if repo_names
        else get_prolog_repos(
            min_stars=min_stars,
            max_stars=max_stars,
            max_repos=max_repos,
            max_pages=max_pages,
            per_page=per_page,
        )
    )

    if not repos:
        print("No repositories found.")
        return []

    return build_repo_snapshots(repos, delay_between=delay_between)


if __name__ == "__main__":
    main()
