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

INSTRUCTION_PROMPT_VERSION = 4
INSTRUCTION_PROMPT_LINES = (
    "Ты пишешь короткое техническое описание файла для разработчика SWI-Prolog.",
    "Опиши назначение файла, ключевые предикаты, входные данные, основное поведение и ожидаемый результат как связный технический текст.",
    "Если имена предикатов помогают точно передать смысл, их можно и нужно упоминать.",
    "Верни один короткий абзац на 1-3 предложения без кода, без markdown, без списков и без заголовков.",
    "Не используй шаблонные заходы и не начинай каждый ответ одинаково. Варьируй первое предложение и точку входа в описание.",
    "Не пиши фразы вроде 'файл содержит следующие функции' и не раскладывай ответ по категориям.",
)
INSTRUCTION_OPENING_STYLE_HINTS = (
    "Начни с назначения файла и его роли в программе.",
    "Начни с того, какое поведение реализуют основные предикаты.",
    "Начни с входных данных и преобразования, которое над ними выполняется.",
    "Начни с логического свойства или результата, который обеспечивает этот код.",
    "Начни с основного предиката, если это делает описание точнее.",
)
INSTRUCTION_FEW_SHOT_EXAMPLES = (
    (
        ":- use_module(library(clpfd)).\n\nlatin_square_row(Row) :-\n    Row ins 1..4,\n    all_distinct(Row).\n\nlatin_square(Board) :-\n    Board = [A, B, C, D],\n    maplist(latin_square_row, Board),\n    transpose(Board, Columns),\n    maplist(all_distinct, Columns).",
        "Файл использует library(clpfd) для описания ограничений над квадратной таблицей и задаёт предикаты latin_square_row/1 и latin_square/1 для проверки корректности строк и столбцов. На вход подаётся структура доски, элементы которой ограничиваются конечными доменами, после чего решение требует попарной различности значений в каждой строке и в каждом столбце.",
    ),
    (
        "token(integer(N)) --> digits(Cs), { number_codes(N, Cs) }.\ntoken(plus) --> \"+\".\nexpr(Value) --> token(integer(A)), token(plus), expr(B), { Value is A + B }.\nexpr(Value) --> token(integer(Value)).\ndigits([C|Cs]) --> [C], { char_type(C, digit) }, digits_rest(Cs).\ndigits_rest([C|Cs]) --> [C], { char_type(C, digit) }, digits_rest(Cs).\ndigits_rest([]) --> [].",
        "Основной предикат expr//1 описывает DCG для разбора простого арифметического выражения и одновременно вычисляет его значение, используя вспомогательные правила token//1, digits//1 и digits_rest//1. Файл принимает последовательность символьных кодов, выделяет целые числа и операторы, а затем рекурсивно строит результат вычисления для суммы.",
    ),
    (
        "reachable(Start, Goal, Path) :-\n    reachable(Start, Goal, [Start], RevPath),\n    reverse(RevPath, Path).\n\nreachable(Node, Node, Visited, Visited).\nreachable(Node, Goal, Visited, Path) :-\n    edge(Node, Next),\n    \\+ memberchk(Next, Visited),\n    reachable(Next, Goal, [Next|Visited], Path).",
        "Предикат reachable/3 ищет путь в графе между начальной и целевой вершиной, накапливая посещённые узлы и исключая повторные обходы через memberchk/2. Внутренний рекурсивный вариант с аккумулятором строит путь в обратном порядке, а внешний предикат преобразует его в итоговый маршрут, пригодный для выдачи пользователю.",
    ),
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
