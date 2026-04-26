import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from github_utils import download_github_repo
from github_prolog_repos import get_prolog_repos
import re
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import subprocess
import json
import tempfile


def is_complex_prolog(content: str, 
                      min_unique_predicates: int = 3,
                      min_rules_ratio: float = 0.05,
                      min_complexity_markers: int = 2
) -> bool:
    """Проверяет, является ли Prolog-код достаточно сложным."""
    code = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
    code = re.sub(r"'[^']*'", '', code)
    
    lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('%')]
    if not lines:
        return False
    
    # Считаем уникальные предикаты
    predicates = set()
    for line in lines:
        match = re.match(r"^([a-z][a-z0-9_]*)\s*\(", line, re.IGNORECASE)
        if match:
            pred_name = match.group(1).lower()
            arity = line.count(',') + 1 if '(' in line else 0
            predicates.add(f"{pred_name}/{arity}")
    
    # Считаем правила и факты
    rules = sum(1 for l in lines if ':-' in l)
    
    # Маркеры сложности
    complexity_markers = 0
    markers = [
        r':-', r'\bvar\(', r'\bfindall\(', r'\bmaplist\(',
        r'\b\+\+', r'->', r';', r'\bis\s+\w', r'\brecursion', r'\b!\b',
    ]
    for marker in markers:
        if re.search(marker, code):
            complexity_markers += 1
    
    if len(predicates) < min_unique_predicates:
        return False
    if rules / max(len(lines), 1) < min_rules_ratio:
        return False
    if complexity_markers < min_complexity_markers:
        return False
    
    return True


def annotate_prolog_with_llm(content: str, max_retries: int = 3) -> str:
    one_shot_prompt = f"""
Ты — технический писатель и экссперт по Swi Prolog. Твоя задача: посмотреть на Prolog-код и написать ТЕХНИЧЕСКОЕ ЗАДАНИЕ (ТЗ), которое заказчик мог бы отправить программисту.
Описывай что нужно сделать, и как это должно быть реализовано. Не упоминай имена предикатов или переменные. Пиши на русском языке, кроме названий функций, атом, предикатов и всего того подобного.
Формат ответа (строго JSON, без markdown):
{{
  "description": "Подробное описание задачи . Что делает код, какие данные обрабатывает, какой результат возвращает, какие методы и алгоритмы используются. Не упоминай имена предикатов или переменные.",
  "data_model": "Какие данные хранятся в системе, формат, структура, отношения между объектами",
  "operations": [
    {{
      "name": "Имя предиката в формате name/arity (например, parent/2)",
      "purpose": "Что делает этот предикат (1 предложение)",
      "input": "Описание входных аргументов с примером",
      "output": "Описание выходных аргументов с примером",
      "logic": "Краткое описание алгоритма: базовый случай, рекурсивный шаг, ключевые конструкции"
    }}
  ],
  }}
Допустимые значения для domain:
- "facts" — факты (предикаты без тела)
- "recursion" — рекурсивные правила
- "arithmetic" — арифметика (is, +, -, *, /)
- "list_processing" — обработка списков [H|T]
- "findall" — сбор решений (findall/3, bagof/3, setof/3)
- "dcg" — грамматика (--> , phrase/2)
- "clpfd" — ограничения (#=, in, labeling)
- "cut" — отсечение (!)
- "negation" — отрицание (\\+, not/1)
- "io" — ввод/вывод (write, format, read)
- "modules" — модули (use_module, module/2)
- "meta" — мета-программирование (call/1, =..)

### ПРИМЕР:

#### Вход:
```
:- use_module(library(lists)).

% Факты: родитель(Родитель, Ребёнок)
parent(alice, bob).
parent(bob, carol).
parent(bob, diana).

% Предикат: предок/2 (рекурсивный)
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% Предикат: все_потомки/2 — собирает всех потомков
all_descendants(X, Descendants) :-
    findall(Y, ancestor(X, Y), Descendants).

% Предикат: поколение/3 — считает поколение между X и Y
generation_count(X, Y, 0) :- parent(X, Y).
generation_count(X, Y, N) :-
    parent(X, Z),
    generation_count(Z, Y, N1),
    N is N1 + 1.
#### Выход:
{{
"description": "Система хранит базу данных о родственных связях между людьми. Каждый факт описывает прямую связь «родитель-ребёнок» между двумя именованными объектами. Система поддерживает четыре операции: проверка родства, поиск всех предков человека, получение списка всех потомков, вычисление количества поколений между двумя родственниками. Все операции работают со структурой где у каждого человека может быть несколько детей и несколько предков.",
"data_model": "Факты хранятся как пары (родитель, ребёнок) где оба элемента — атомы (имена). Пример: parent(alice, bob). Один человек может быть родителем нескольких детей. Один человек может иметь нескольких предков через цепочку родительских связей. Связь направлена от родителя к ребёнку.",
"operations": [
{{
"name": "parent/2",
"purpose": "Хранит факты о прямых родительских связях",
"input": "Два атома-имени: родитель и ребёнок",
"output": "Истина если связь существует в базе фактов",
"logic": "Факт без тела: проверяется наличие записи в базе данных"
}},
{{
"name": "ancestor/2",
"purpose": "Проверяет является ли первый аргумент предком второго (прямым или через цепочку)",
"input": "Два атома-имени: возможный предок и возможный потомок",
"output": "Истина если связь предка существует",
"logic": "Базовый случай: прямая связь через parent/2. Рекурсивный случай: найти промежуточного Z где первый — родитель Z, а Z — предок второго"
}},
{{
"name": "all_descendants/2",
"purpose": "Собирает всех потомков заданного человека в список",
"input": "Атом-имя человека",
"output": "Список атомов-имён всех потомков: [bob, carol, diana]",
"logic": "Использует findall/3 для сбора всех Y где ancestor(человек, Y) истинно"
}},
{{
"name": "generation_count/3",
"purpose": "Вычисляет количество поколений (шагов) между предком и потомком",
"input": "Два атома-имени: предок и потомок",
"output": "Целое число: 0 для прямого родителя, 1 для дедушки и т.д.",
"logic": "Базовый случай: parent/2 → N=0. Рекурсивный случай: рекурсивный вызов с промежуточным Z, затем арифметика N is N1 + 1"
}}
],
}}
#### Файл который нужно обработать:
{content}"""
    model_name = os.getenv("OLLAMA_MODEL", "llama3")  # По умолчанию llama3

    for attempt in range(max_retries):
        try:
            # Создаем временный файл с промптом
            url = "http://localhost:11434/api/generate"
            
            payload = {
                "model": model_name,
                "prompt": one_shot_prompt,
                "stream": False,
                
            }
            
            response = requests.post(
                url,
                json=payload,
                timeout=120,  # 2 минуты таймаут
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                annotation = result.get('response', '').strip()
                
                if annotation:
                    # Очищаем вывод
                    if "```" in annotation:
                        annotation = annotation.replace("```prolog", "").replace("```", "").strip()
                    return annotation
                else:
                    print("   ⚠️ Пустой ответ от LLM")
            else:
                print(f"   ⚠️ Ошибка HTTP {response.status_code}: {response.text}")
            
            if attempt < max_retries - 1:
                    time.sleep(10)
                
        except subprocess.TimeoutExpired:
            print(f"Таймаут ollama (попытка {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(5)
        except FileNotFoundError:
            print("❌ Команда 'ollama' не найдена. Убедитесь, что Ollama установлена и добавлена в PATH")
            return ""
        except Exception as e:
            print(f"Ошибка выполнения ollama (попытка {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    
    return ""

def is_prolog_compilable(file_path: Path, timeout: int = 15) -> bool:
    
    swipl = shutil.which('swipl')
    if not swipl:
        print("   ⚠️ SWI-Prolog не найден в PATH. Проверка компилируемости пропущена.")
        return True  
    try:
        
        cmd = [swipl, '-q', '--no-autoload', '-c', str(file_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        # SWI-Prolog создаёт .qlf файл при успешной компиляции. Удаляем его, чтобы не засорять диск.
        qlf_path = file_path.with_suffix('.qlf')
        if qlf_path.exists():
            qlf_path.unlink(missing_ok=True)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"   ⏱️ Превышен таймаут компиляции для {file_path.name}")
        return False
    except Exception as e:
        print(f"   ❌ Ошибка проверки компилируемости {file_path.name}: {e}")
        return False


import json
import time
from pathlib import Path
from typing import Tuple, Dict

def process_prolog_file(file_path: Path, repo_path: str, repo_name: str) -> Tuple[bool, Dict]:
    try:
        # Пропускаем большие файлы
        try:
            if file_path.stat().st_size > 100_000:
                print(f"   ⏩ Пропускаем слишком большой файл: {file_path.name}")
                return False, {}
        except OSError:
            pass

        if not is_prolog_compilable(file_path):
            print(f"   ❌ Файл {file_path.name} содержит ошибки компиляции. Пропускаем.")
            return False, {}

        # 2. Читаем содержимое файла
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 3. Проверяем сложность
        if not is_complex_prolog(content):
            return False, {}

        print(f"   🤖 Аннотируем с контекстом: {file_path.name}")

        
        repo_context = {}
        readme_content = ""
        for readme_name in ['README.md', 'README.txt', 'README', 'readme.md']:
            readme_path = Path(repo_path) / readme_name
            if readme_path.exists() and readme_path.is_file():
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        readme_content = f.read(1500)
                    repo_context['README'] = readme_content + "..." if len(readme_content) == 1500 else readme_content
                    break
                except Exception:
                    continue

        # 5. Формируем контекст
        context_str = content
        if 'README' in repo_context:
            context_str += "\n\nКОНТЕКСТ РЕПОЗИТОРИЯ:\n\n" + repo_context['README']

        # 6. Запрос аннотации с ретраями и валидацией JSON
        annotation = None
        max_retries = 1
        for attempt in range(1, max_retries + 1):
            try:
                raw_annotation = annotate_prolog_with_llm(context_str, 3)
                if not raw_annotation:
                    print(f"   ⚠️ Попытка {attempt}/{max_retries}: Пустой ответ от LLM.")
                    continue

                # Очистка от markdown-обёрток (```json ... ```)
                clean_text = raw_annotation.strip()
                if clean_text.startswith("```"):
                    parts = clean_text.split("```")
                    if len(parts) >= 3:
                        clean_text = parts[1].strip()
                        if clean_text.lower().startswith("json"):
                            clean_text = clean_text[4:].strip()
                    elif len(parts) == 2:
                        clean_text = parts[1].strip()

                # Валидация JSON
                json.loads(clean_text)
                annotation = clean_text
                print(f"   ✅ JSON валиден с {attempt} попытки.")
                break

            except json.JSONDecodeError as e:
                print(f"   ⚠️ Попытка {attempt}/{max_retries}: Невалидный JSON. Ошибка: {e}")
            except Exception as e:
                print(f"   ⚠️ Попытка {attempt}/{max_retries}: Ошибка вызова LLM: {e}")

            if attempt < max_retries:
                time.sleep(1.5)  # Экспоненциальная задержка снижает нагрузку на API

        if not annotation:
            annotation = annotate_prolog_with_llm(context_str, 3)

        file_info = {
            'path': str(file_path),
            'filename': file_path.name,
            'annotation': annotation,
            'content_preview': content,
        }

        return True, file_info

    except Exception as e:
        print(f"   ⚠️ Ошибка обработки файла {file_path}: {e}")
        return False, {}
def find_prolog_files(directory: str) -> List[Path]:
    """Находит все Prolog-файлы в директории и поддиректориях."""
    prolog_extensions = {'.pl'}
    prolog_files = []
    
    for root, dirs, files in os.walk(directory):
        # Пропускаем служебные папки
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'node_modules'}]
        
        for file in files:
            if Path(file).suffix.lower() in prolog_extensions:
                prolog_files.append(Path(root) / file)
    
    return prolog_files