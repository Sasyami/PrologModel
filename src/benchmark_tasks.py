from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkTestCase:
    goal: str
    expected_stdout: str
    timeout_sec: int = 5


@dataclass
class BenchmarkTask:
    task_id: str
    prompt: str
    reference_solution: str | None = None
    tests: list[BenchmarkTestCase] | None = None


def load_benchmark_tasks(path: Path) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            payload = json.loads(line)
            tests = [
                BenchmarkTestCase(
                    goal=test["goal"],
                    expected_stdout=test["expected_stdout"],
                    timeout_sec=int(test.get("timeout_sec", 5)),
                )
                for test in payload.get("tests", [])
            ]

            tasks.append(
                BenchmarkTask(
                    task_id=payload["id"],
                    prompt=payload["prompt"],
                    reference_solution=payload.get("reference_solution"),
                    tests=tests,
                )
            )

    return tasks
