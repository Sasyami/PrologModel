# PrologModel

Проект для сборки корпуса Prolog-кода, генерации описаний, дообучения модели и проверки качества на SWI-Prolog benchmark.

## Что умеет проект

- скачивает и фильтрует Prolog-репозитории с GitHub;
- сохраняет компактные снапшоты кода в `data/repos` и `data/annotated_repos`;
- удаляет похожие файлы по коду до генерации описаний;
- генерирует file-level описания через Ollama;
- готовит датасет для SFT;
- дообучает модель в `src/model.ipynb`;
- проверяет качество через `swipl` и считает `pass@k`.

## Структура

- `benchmark/tasks_swipl.jsonl` — benchmark-задачи для SWI-Prolog.
- `data/build_repo_snapshots.py` — скачивание репозиториев и первичный фильтр Prolog-файлов.
- `data/cleanup_similar_code.py` — удаление exact и semantic дубликатов по коду.
- `data/generate_descriptions.py` — генерация описаний и dedup по описаниям.
- `data/pipeline_constants.py` — обязательные настройки из `.env`.
- `src/dataset_loader.py` — загрузка train-корпуса из `annotated_repos`.
- `src/model.ipynb` — notebook для обучения и smoke-test модели.
- `src/generate_solutions.py` — генерация решений benchmark-задач через Ollama.
- `src/eval_runner.py` — проверка решений через `swipl`.
- `src/pass_at_k.py` — расчёт `pass@k`.
- `tests/test_pass_at_k.py` — текущие автотесты.

## Требования

- Python 3.10+
- SWI-Prolog в `PATH`
- Ollama для генерации описаний и benchmark-решений
- NVIDIA GPU для обучения желательно, но не обязательно

Зависимости:

```powershell
pip install -r requirements.txt
```

## Конфигурация

Проект читает обязательные параметры из `.env`.

Ключевые настройки:

- `TRAIN_BASE_MODEL_ID`
- `OLLAMA_MODEL`
- `OLLAMA_GENERATE_URL`
- `GITHUB_TOKEN`
- `GITHUB_MIN_STARS`
- `GITHUB_MAX_STARS`
- `GITHUB_MAX_REPOS`
- `MAX_STRIPPED_CODE_TOKENS`
- `MAX_TRAIN_SAMPLE_TOKENS`
- `DEDUP_COSINE_THRESHOLD`
- `CODE_CLEANUP_MODEL_ID`
- `CODE_CLEANUP_COSINE_THRESHOLD`

Если нужной переменной нет, проект падает сразу и просит настроить её явно.

## Типичный порядок работы

### 1. Собрать снапшоты репозиториев

```powershell
python -c "from data.build_repo_snapshots import main; main()"
```

Этот шаг:

- ищет Prolog-репозитории на GitHub;
- скачивает их;
- проверяет файлы через compile-check и фильтры;
- сохраняет исходные `.pl` в `data/repos`;
- сохраняет очищенные `.pl` и `.json` в `data/annotated_repos`.

### 2. Удалить похожие файлы по коду

```powershell
python -c "from data.cleanup_similar_code import main; main()"
```

Этот шаг:

- удаляет exact duplicates по hash кода;
- удаляет semantic duplicates по embeddings кода;
- удаляет слишком длинные файлы по токенам train-модели.

### 3. Сгенерировать описания

```powershell
python -c "from data.generate_descriptions import main; main()"
```

Этот шаг:

- читает очищенный код;
- строит описание файла через Ollama;
- сохраняет `.txt` рядом с `.pl` и `.json`;
- удаляет дубликаты по embeddings описаний.

### 4. Обучить модель

Открой:

- `src/model.ipynb`

Notebook сейчас:

- грузит train-модель из `TRAIN_BASE_MODEL_ID`;
- готовит QLoRA/4-bit загрузку;
- делает smoke-test необученной модели;
- строит датасет из `annotated_repos`;
- учит модель только на токенах ответа `assistant`.

### 5. Проверить benchmark

Reference-check:

```powershell
python -c "from src.eval_runner import main; main(mode='reference')"
```

Сгенерировать решения через Ollama:

```powershell
python -c "from src.generate_solutions import main; main()"
```

Проверить решения:

```powershell
python -c "from src.eval_runner import main; main(mode='solutions', solutions_dir='outputs/solutions')"
```

## Pass@k

Для `pass@k` в проекте есть отдельный модуль:

- `src/pass_at_k.py`

Текущие тесты:

```powershell
python -m unittest tests.test_pass_at_k
```

## Что важно помнить

- `data/repos` и `data/annotated_repos` — это локальные корпусные артефакты, а не исходники проекта.
- Большие модели, `.gguf`, merged checkpoints и временные outputs лучше держать локально и не коммитить.
- Для `Qwen3.5` в notebook уже выключен `thinking`-режим через `enable_thinking=False`.

## Текущее состояние

Сейчас репозиторий содержит:

- корпусный пайплайн;
- benchmark на SWI-Prolog;
- notebook для обучения;
- тесты только на `pass@k`.

README описывает именно текущую структуру проекта, а не старую версию, где был только baseline benchmark.
