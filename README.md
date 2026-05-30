# SWI-Prolog LLM Baseline Benchmark

Small benchmark to check whether an LLM has issues solving SWI-Prolog tasks before any fine-tuning.

## What this project does

- Stores SWI-Prolog tasks in JSONL format
- Runs solutions through `swipl`
- Reports baseline metrics:
  - `pass_rate`
  - `syntax_pass_rate`
  - `timeout_rate`

## Project layout

- `benchmark/tasks_swipl.jsonl` - task dataset
- `src/eval_runner.py` - benchmark runner
- `outputs/` - optional model-generated `.pl` files (one per task id)

## Prerequisites

- Python 3.10+
- SWI-Prolog installed and available as `swipl`

## Run sanity check on reference solutions

```powershell
py -m src.eval_runner --tasks benchmark/tasks_swipl.jsonl --mode reference
```

This verifies that tasks/tests are valid.

## Run baseline on model outputs

Put generated files into `outputs/solutions/` with names `<task_id>.pl`, then run:

```powershell
py -m src.eval_runner --tasks benchmark/tasks_swipl.jsonl --mode solutions --solutions-dir outputs/solutions
```

## Auto-generate solutions with open LLM (Ollama)

1) Install Ollama and pull a model:

```powershell
ollama pull qwen2.5-coder:7b
```

2) Generate `.pl` files for all tasks:

```powershell
py -m src.generate_solutions --tasks benchmark/tasks_swipl.jsonl --model qwen2.5-coder:7b --out-dir outputs/solutions
```

3) Evaluate generated solutions:

```powershell
py -m src.eval_runner --tasks benchmark/tasks_swipl.jsonl --mode solutions --solutions-dir outputs/solutions
```

## Task format (JSONL)

Each line includes:

- `id`: unique task id
- `prompt`: what you send to the model
- `reference_solution`: known-good SWI-Prolog solution
- `tests`: list of checks
  - `goal`: Prolog goal executed via `-g`
  - `expected_stdout`: exact expected stdout
  - `timeout_sec`: optional timeout per test

