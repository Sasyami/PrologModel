from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TestCase:
    goal: str
    expected_stdout: str
    timeout_sec: int = 5


@dataclass
class Task:
    task_id: str
    prompt: str
    reference_solution: str
    tests: list[TestCase]


def load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tests = [
                TestCase(
                    goal=t["goal"],
                    expected_stdout=t["expected_stdout"],
                    timeout_sec=int(t.get("timeout_sec", 5)),
                )
                for t in obj["tests"]
            ]
            tasks.append(
                Task(
                    task_id=obj["id"],
                    prompt=obj["prompt"],
                    reference_solution=obj["reference_solution"],
                    tests=tests,
                )
            )
    return tasks


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
        raise SystemExit(
            "SWI-Prolog executable `swipl` not found in PATH. Install SWI-Prolog first."
        )


def evaluate_task(task: Task, solution_text: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="swipl_eval_") as td:
        solution_path = Path(td) / "solution.pl"
        solution_path.write_text(solution_text, encoding="utf-8")

        # Syntax/load check
        _, rc, timeout = run_swipl(solution_path, "true", timeout_sec=5)
        syntax_ok = (rc == 0) and not timeout

        passed = 0
        timeouts = 0
        for test in task.tests:
            out, rc, tmo = run_swipl(solution_path, test.goal, timeout_sec=test.timeout_sec)
            if tmo:
                timeouts += 1
                continue
            if rc == 0 and out == test.expected_stdout:
                passed += 1

        all_passed = passed == len(task.tests)
        return {
            "task_id": task.task_id,
            "syntax_ok": syntax_ok,
            "passed_tests": passed,
            "total_tests": len(task.tests),
            "all_passed": all_passed,
            "timeouts": timeouts,
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate SWI-Prolog tasks for LLM baseline")
    p.add_argument("--tasks", required=True, help="Path to JSONL tasks")
    p.add_argument(
        "--mode",
        choices=["reference", "solutions"],
        required=True,
        help="reference: evaluate reference_solution, solutions: evaluate files from --solutions-dir",
    )
    p.add_argument(
        "--solutions-dir",
        default=None,
        help="Directory with <task_id>.pl files (required in solutions mode)",
    )
    args = p.parse_args(argv)

    check_swipl_exists()
    tasks = load_tasks(Path(args.tasks))
    if not tasks:
        raise SystemExit("No tasks found.")

    solutions_dir = Path(args.solutions_dir) if args.solutions_dir else None
    if args.mode == "solutions" and not solutions_dir:
        raise SystemExit("--solutions-dir is required in solutions mode.")

    results: list[dict[str, Any]] = []
    for task in tasks:
        if args.mode == "reference":
            solution_text = task.reference_solution
        else:
            solution_file = solutions_dir / f"{task.task_id}.pl"
            if not solution_file.exists():
                results.append(
                    {
                        "task_id": task.task_id,
                        "syntax_ok": False,
                        "passed_tests": 0,
                        "total_tests": len(task.tests),
                        "all_passed": False,
                        "timeouts": 0,
                        "missing_solution": True,
                    }
                )
                continue
            solution_text = solution_file.read_text(encoding="utf-8")

        results.append(evaluate_task(task, solution_text))

    total = len(results)
    syntax_ok = sum(1 for r in results if r.get("syntax_ok"))
    passed_all = sum(1 for r in results if r.get("all_passed"))
    total_timeouts = sum(int(r.get("timeouts", 0)) for r in results)
    total_tests = sum(int(r.get("total_tests", 0)) for r in results)

    print("=== SWI-Prolog Baseline Report ===")
    print(f"tasks: {total}")
    print(f"pass_rate: {passed_all / total:.3f} ({passed_all}/{total})")
    print(f"syntax_pass_rate: {syntax_ok / total:.3f} ({syntax_ok}/{total})")
    timeout_rate = (total_timeouts / total_tests) if total_tests else 0.0
    print(f"timeout_rate: {timeout_rate:.3f} ({total_timeouts}/{total_tests})")
    print("")
    print("Per-task:")
    for r in results:
        status = "PASS" if r["all_passed"] else "FAIL"
        print(
            f"- {r['task_id']}: {status}, "
            f"syntax_ok={r['syntax_ok']}, tests={r['passed_tests']}/{r['total_tests']}, "
            f"timeouts={r.get('timeouts', 0)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

