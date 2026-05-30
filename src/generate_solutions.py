from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark_tasks import BenchmarkTask, load_benchmark_tasks


def build_model_prompt(task: BenchmarkTask) -> str:
    return (
        "Ты пишешь решение на SWI-Prolog.\n"
        "Верни только код программы, без пояснений, без markdown.\n"
        "Код должен быть совместим со SWI-Prolog и запускаться через swipl.\n\n"
        f"Задача:\n{task.prompt}\n"
    )


def extract_code(text: str) -> str:
    match = re.search(r"```(?:prolog)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


def generate_with_ollama(model: str, prompt: str, timeout_sec: int = 120) -> str:
    cmd = ["ollama", "run", model, prompt]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ollama failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


def main(
    tasks_path: str = "benchmark/tasks_swipl.jsonl",
    out_dir: str = "outputs/solutions",
    model: str = "qwen2.5-coder:7b",
    overwrite: bool = False,
    sleep_ms: int = 0,
) -> int:
    tasks = load_benchmark_tasks(Path(tasks_path))
    if not tasks:
        print("No tasks found.")
        return 1

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped = 0
    failed = 0

    for task in tasks:
        out_file = out_dir_path / f"{task.task_id}.pl"
        if out_file.exists() and not overwrite:
            skipped += 1
            print(f"[skip] {task.task_id} (already exists)")
            continue

        prompt = build_model_prompt(task)
        print(f"[gen ] {task.task_id}")
        try:
            raw = generate_with_ollama(model=model, prompt=prompt)
            code = extract_code(raw)
            out_file.write_text(code, encoding="utf-8")
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"[fail] {task.task_id}: {exc}")

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    print("")
    print("=== Generation summary ===")
    print(f"generated: {ok}")
    print(f"skipped: {skipped}")
    print(f"failed: {failed}")
    print(f"out_dir: {out_dir_path}")
    return 0


if __name__ == "__main__":
    main()
