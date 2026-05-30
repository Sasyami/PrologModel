from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark_tasks import BenchmarkTask, load_benchmark_tasks


def run_swipl(solution_path: Path, goal: str, timeout_sec: int) -> tuple[str, int, bool]:
    cmd = [
        "swipl",
        "-q",
        "-s",
        str(solution_path),
        "-g",
        goal,
        "-t",
        "halt",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", -1, True

    out = (proc.stdout or "").strip()
    return out, proc.returncode, False


def check_swipl_exists() -> None:
    if shutil.which("swipl") is None:
        raise RuntimeError("SWI-Prolog executable `swipl` not found in PATH.")


def evaluate_task(task: BenchmarkTask, solution_text: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="swipl_eval_") as td:
        solution_path = Path(td) / "solution.pl"
        solution_path.write_text(solution_text, encoding="utf-8")

        _, rc, timeout = run_swipl(solution_path, "true", timeout_sec=5)
        syntax_ok = (rc == 0) and not timeout

        tests = task.tests or []
        passed = 0
        timeouts = 0
        for test in tests:
            out, rc, tmo = run_swipl(solution_path, test.goal, timeout_sec=test.timeout_sec)
            if tmo:
                timeouts += 1
                continue
            if rc == 0 and out == test.expected_stdout:
                passed += 1

        all_passed = passed == len(tests)
        return {
            "task_id": task.task_id,
            "syntax_ok": syntax_ok,
            "passed_tests": passed,
            "total_tests": len(tests),
            "all_passed": all_passed,
            "timeouts": timeouts,
        }


def main(
    tasks_path: str = "benchmark/tasks_swipl.jsonl",
    mode: str = "reference",
    solutions_dir: str | None = None,
) -> int:
    check_swipl_exists()
    tasks = load_benchmark_tasks(Path(tasks_path))
    if not tasks:
        print("No tasks found.")
        return 1

    solutions_dir_path = Path(solutions_dir) if solutions_dir else None
    if mode == "solutions" and not solutions_dir_path:
        print("solutions_dir is required in solutions mode.")
        return 1

    results: list[dict[str, Any]] = []
    for task in tasks:
        if mode == "reference":
            solution_text = task.reference_solution or ""
        else:
            solution_file = solutions_dir_path / f"{task.task_id}.pl"
            if not solution_file.exists():
                results.append(
                    {
                        "task_id": task.task_id,
                        "syntax_ok": False,
                        "passed_tests": 0,
                        "total_tests": len(task.tests or []),
                        "all_passed": False,
                        "timeouts": 0,
                        "missing_solution": True,
                    }
                )
                continue
            solution_text = solution_file.read_text(encoding="utf-8")

        results.append(evaluate_task(task, solution_text))

    total = len(results)
    syntax_ok = sum(1 for result in results if result.get("syntax_ok"))
    passed_all = sum(1 for result in results if result.get("all_passed"))
    total_timeouts = sum(int(result.get("timeouts", 0)) for result in results)
    total_tests = sum(int(result.get("total_tests", 0)) for result in results)

    print("=== SWI-Prolog Baseline Report ===")
    print(f"tasks: {total}")
    print(f"pass_rate: {passed_all / total:.3f} ({passed_all}/{total})")
    print(f"syntax_pass_rate: {syntax_ok / total:.3f} ({syntax_ok}/{total})")
    timeout_rate = (total_timeouts / total_tests) if total_tests else 0.0
    print(f"timeout_rate: {timeout_rate:.3f} ({total_timeouts}/{total_tests})")
    print("")
    print("Per-task:")
    for result in results:
        status = "PASS" if result["all_passed"] else "FAIL"
        print(
            f"- {result['task_id']}: {status}, "
            f"syntax_ok={result['syntax_ok']}, tests={result['passed_tests']}/{result['total_tests']}, "
            f"timeouts={result.get('timeouts', 0)}"
        )

    return 0


if __name__ == "__main__":
    main()
