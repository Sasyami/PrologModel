from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from src.benchmark_tasks import BenchmarkTask, BenchmarkTestCase, load_benchmark_tasks
from src.eval_runner import evaluate_task
from src.pass_at_k import (
    estimate_mean_pass_at_k,
    estimate_task_pass_at_k,
)


def _make_double_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="double_number",
        prompt="Implement double/2.",
        tests=[
            BenchmarkTestCase(
                goal="double(4,X), writeln(X)",
                expected_stdout="8",
            ),
            BenchmarkTestCase(
                goal="double(0,X), writeln(X)",
                expected_stdout="0",
            ),
        ],
    )


def _make_member_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="my_member",
        prompt="Implement my_member/2 without using member/2.",
        tests=[
            BenchmarkTestCase(
                goal="(my_member(2,[1,2,3]) -> writeln(true) ; writeln(false))",
                expected_stdout="true",
            ),
            BenchmarkTestCase(
                goal="(my_member(9,[1,2,3]) -> writeln(true) ; writeln(false))",
                expected_stdout="false",
            ),
        ],
    )


@unittest.skipUnless(shutil.which("swipl"), "SWI-Prolog is required for integration tests")
class PassAtKPrologIntegrationTests(unittest.TestCase):
    def test_explicit_prolog_goals_are_visible_in_double_task(self) -> None:
        task = _make_double_task()
        self.assertEqual(
            [test.goal for test in task.tests or []],
            [
                "double(4,X), writeln(X)",
                "double(0,X), writeln(X)",
            ],
        )

    def test_explicit_prolog_goals_are_visible_in_member_task(self) -> None:
        task = _make_member_task()
        self.assertEqual(
            [test.goal for test in task.tests or []],
            [
                "(my_member(2,[1,2,3]) -> writeln(true) ; writeln(false))",
                "(my_member(9,[1,2,3]) -> writeln(true) ; writeln(false))",
            ],
        )

    def test_evaluate_task_reports_partial_pass_counts(self) -> None:
        task = _make_double_task()
        solution = "double(X,Y) :- X > 0, Y is X * 2.\n"
        result = evaluate_task(task, solution)

        self.assertTrue(result["syntax_ok"])
        self.assertEqual(result["passed_tests"], 1)
        self.assertEqual(result["total_tests"], 2)
        self.assertFalse(result["all_passed"])
        self.assertEqual(result["timeouts"], 0)

    def test_invalid_solution_does_not_pass_task(self) -> None:
        task = _make_double_task()
        solution = ":- use_module(library(clpfd).\n"
        result = evaluate_task(task, solution)

        self.assertEqual(result["passed_tests"], 0)
        self.assertFalse(result["all_passed"])

    def test_pass_at_k_from_real_prolog_candidates(self) -> None:
        task = _make_double_task()

        solutions = [
            "double(X,Y) :- Y is X * 2.\n",
            "double(X,Y) :- X > 0, Y is X * 2.\n",
            "double(X,Y :- Y is X * 2.\n",
        ]

        outcomes = [evaluate_task(task, solution)["all_passed"] for solution in solutions]

        self.assertEqual(outcomes, [True, False, False])
        self.assertAlmostEqual(estimate_task_pass_at_k(outcomes, k=1), 1 / 3)
        self.assertAlmostEqual(estimate_task_pass_at_k(outcomes, k=2), 2 / 3)
        self.assertEqual(estimate_task_pass_at_k(outcomes, k=3), 1.0)

    def test_pass_at_k_for_member_task_candidates(self) -> None:
        task = _make_member_task()
        solutions = [
            "my_member(X,[X|_]).\nmy_member(X,[_|T]) :- my_member(X,T).\n",
            "my_member(X,[X|_]).\n",
            "my_member(X,Y :- fail.\n",
        ]

        outcomes = [evaluate_task(task, solution)["all_passed"] for solution in solutions]

        self.assertEqual(outcomes, [True, False, False])
        self.assertAlmostEqual(estimate_task_pass_at_k(outcomes, k=1), 1 / 3)
        self.assertAlmostEqual(estimate_task_pass_at_k(outcomes, k=2), 2 / 3)

    def test_mean_pass_at_k_across_two_real_prolog_tasks(self) -> None:
        double_outcomes = [
            evaluate_task(_make_double_task(), "double(X,Y) :- Y is X * 2.\n")["all_passed"],
            evaluate_task(_make_double_task(), "double(X,Y) :- X > 0, Y is X * 2.\n")[
                "all_passed"
            ],
            evaluate_task(_make_double_task(), "double(X,Y :- Y is X * 2.\n")["all_passed"],
        ]
        member_outcomes = [
            evaluate_task(
                _make_member_task(),
                "my_member(X,[X|_]).\nmy_member(X,[_|T]) :- my_member(X,T).\n",
            )["all_passed"],
            evaluate_task(_make_member_task(), "my_member(X,[X|_]).\n")["all_passed"],
            evaluate_task(_make_member_task(), "my_member(X,Y :- fail.\n")["all_passed"],
        ]

        self.assertAlmostEqual(estimate_mean_pass_at_k([double_outcomes, member_outcomes], k=1), 1 / 3)
        self.assertAlmostEqual(estimate_mean_pass_at_k([double_outcomes, member_outcomes], k=2), 2 / 3)

    def test_pass_at_k_is_monotonic_across_real_prolog_outcomes(self) -> None:
        task = _make_member_task()
        outcomes = [
            evaluate_task(
                task,
                "my_member(X,[X|_]).\nmy_member(X,[_|T]) :- my_member(X,T).\n",
            )["all_passed"],
            evaluate_task(task, "my_member(X,[X|_]).\n")["all_passed"],
            evaluate_task(task, "my_member(X,Y :- fail.\n")["all_passed"],
        ]

        self.assertLessEqual(
            estimate_task_pass_at_k(outcomes, k=1),
            estimate_task_pass_at_k(outcomes, k=2),
        )
        self.assertLessEqual(
            estimate_task_pass_at_k(outcomes, k=2),
            estimate_task_pass_at_k(outcomes, k=3),
        )

    def test_reference_solution_from_benchmark_passes(self) -> None:
        tasks = load_benchmark_tasks(Path("benchmark/tasks_swipl.jsonl"))
        task = next(task for task in tasks if task.task_id == "my_member")

        self.assertEqual(
            [test.goal for test in task.tests or []],
            [
                "(my_member(2,[1,2,3]) -> writeln(true) ; writeln(false))",
                "(my_member(9,[1,2,3]) -> writeln(true) ; writeln(false))",
            ],
        )

        result = evaluate_task(task, task.reference_solution or "")
        self.assertTrue(result["syntax_ok"])
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["passed_tests"], 2)


if __name__ == "__main__":
    unittest.main()
