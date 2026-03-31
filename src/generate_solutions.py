from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Task:
    task_id: str
    prompt: str


def load_tasks(path: Path) -> list[Task]:
    out: list[Task] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(Task(task_id=obj["id"], prompt=obj["prompt"]))
    return out


def build_model_prompt(task: Task) -> str:
    return (
        "Ты пишешь решение на SWI-Prolog.\n"
        "Верни только код программы, без пояснений, без markdown.\n"
        "Код должен быть совместим со SWI-Prolog и запускаться через swipl.\n\n"
        f"Задача:\n{task.prompt}\n"
    )


def extract_code(text: str) -> str:
    # If model still returns markdown, extract first fenced block.
    m = re.search(r"```(?:prolog)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip() + "\n"
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate SWI-Prolog solutions with an open LLM (Ollama).")
    p.add_argument("--tasks", required=True, help="Path to tasks JSONL")
    p.add_argument("--out-dir", default="outputs/solutions", help="Where to save <task_id>.pl")
    p.add_argument("--model", default="qwen2.5-coder:7b", help="Ollama model name")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .pl files")
    p.add_argument("--sleep-ms", type=int, default=0, help="Delay between generations")
    args = p.parse_args(argv)

    tasks = load_tasks(Path(args.tasks))
    if not tasks:
        raise SystemExit("No tasks found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped = 0
    failed = 0

    for task in tasks:
        out_file = out_dir / f"{task.task_id}.pl"
        if out_file.exists() and not args.overwrite:
            skipped += 1
            print(f"[skip] {task.task_id} (already exists)")
            continue

        prompt = build_model_prompt(task)
        print(f"[gen ] {task.task_id}")
        try:
            raw = generate_with_ollama(model=args.model, prompt=prompt)
            code = extract_code(raw)
            out_file.write_text(code, encoding="utf-8")
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[fail] {task.task_id}: {e}")

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print("")
    print("=== Generation summary ===")
    print(f"generated: {ok}")
    print(f"skipped: {skipped}")
    print(f"failed: {failed}")
    print(f"out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

