from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")
else:
    _load_env_file(PROJECT_ROOT / ".env")


def _get_required(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_required_int(name: str) -> int:
    return int(_get_required(name))


def _get_required_float(name: str) -> float:
    return float(_get_required(name))


REPOS_DIR = DATA_ROOT / "repos"
ANNOTATED_REPOS_DIR = DATA_ROOT / "annotated_repos"
DUPLICATE_LOG_PATH = DATA_ROOT / "duplicate_log.jsonl"

README_CANDIDATES = ("README.md", "README.txt", "README", "readme.md")
IGNORED_DIR_NAMES = {".git", "__pycache__", ".venv", "node_modules"}

TRAIN_BASE_MODEL_ID = _get_required("TRAIN_BASE_MODEL_ID")
OLLAMA_MODEL = _get_required("OLLAMA_MODEL")
OLLAMA_GENERATE_URL = _get_required("OLLAMA_GENERATE_URL")
GITHUB_SEARCH_URL = _get_required("GITHUB_SEARCH_URL")

INSTRUCTION_PROMPT_VERSION = 2
INSTRUCTION_PROMPT_LINES = (
    "Ты пишешь техническое описание задачи для разработчика SWI-Prolog.",
    "Опиши назначение файла, входные данные, ключевое поведение и ожидаемый результат как связное описание.",
    "Верни один короткий технический текст без списков, без подпунктов, без нумерации, без маркированных перечислений и без заголовков.",
    "Пиши сплошным абзацем на 1-3 предложений. Не используй формулировки вида 'следующие функции' и не раскладывай ответ по категориям.",
    "Не пиши код. Не используй markdown. Не упоминай имена предикатов, если можно обойтись без них.",
)
INSTRUCTION_README_CONTEXT_LINE = (
    "Контекст README репозитория. Используй его только если он помогает понять задачу:"
)

COMPILE_TIMEOUT_SEC = _get_required_int("COMPILE_TIMEOUT_SEC")
DOWNLOAD_DELAY_SEC = _get_required_float("DOWNLOAD_DELAY_SEC")
LLM_TIMEOUT_SEC = _get_required_int("LLM_TIMEOUT_SEC")
GITHUB_SEARCH_TIMEOUT_SEC = _get_required_int("GITHUB_SEARCH_TIMEOUT_SEC")
GITHUB_RATE_LIMIT_DELAY_SEC = _get_required_int("GITHUB_RATE_LIMIT_DELAY_SEC")
GITHUB_PAGE_DELAY_SEC = _get_required_int("GITHUB_PAGE_DELAY_SEC")
GITHUB_SEARCH_RESULT_LIMIT = _get_required_int("GITHUB_SEARCH_RESULT_LIMIT")
GITHUB_MIN_STARS = _get_required_int("GITHUB_MIN_STARS")
GITHUB_MAX_STARS = _get_required_int("GITHUB_MAX_STARS")
GITHUB_MIN_STAR_BUCKET_SIZE = _get_required_int("GITHUB_MIN_STAR_BUCKET_SIZE")
GITHUB_MAX_REPOS = _get_required_int("GITHUB_MAX_REPOS")
MAX_README_PROMPT_CHARS = _get_required_int("MAX_README_PROMPT_CHARS")

MIN_COMPLEXITY_SCORE = _get_required_int("MIN_COMPLEXITY_SCORE")
MAX_STRIPPED_CODE_TOKENS = _get_required_int("MAX_STRIPPED_CODE_TOKENS")
MAX_TRAIN_SAMPLE_TOKENS = _get_required_int("MAX_TRAIN_SAMPLE_TOKENS")
DEDUP_COSINE_THRESHOLD = _get_required_float("DEDUP_COSINE_THRESHOLD")
CODE_CLEANUP_MODEL_ID = _get_required("CODE_CLEANUP_MODEL_ID")
CODE_CLEANUP_COSINE_THRESHOLD = _get_required_float("CODE_CLEANUP_COSINE_THRESHOLD")
CODE_CLEANUP_MAX_TOKENS = _get_required_int("CODE_CLEANUP_MAX_TOKENS")

TRAINING_SYSTEM_PROMPT = (
    "Вы - экспертный разработчик на Prolog. "
    "Генерируйте чистый, идиоматичный код на Prolog на основе описания."
)
