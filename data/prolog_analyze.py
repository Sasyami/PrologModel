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

@DeprecationWarning
def annotate_prolog_with_llm(content: str, max_retries: int = 2) -> str:
    """
    Аннотирует Prolog-код с помощью LLM через системную команду ollama.
    
    Returns:
        str: Аннотация кода или пустая строка при ошибке
    """
    # Ограничиваем размер контента
    content_preview = content
    
    prompt = f"""Ты — эксперт по SWI-Prolog 9.x. Твоя задача: проанализировать Prolog-файл и вернуть структурированную аннотацию в формате JSON.

Формат ответа (строго JSON, без markdown):
{
  "task": "Подробное описание задачи (3-5 предложений). Что делает код, какие данные обрабатывает, какой результат возвращает.",
  "domain": "область применения (списки, графы, семья, парсинг, арифметика...)",
  "imports": ["library(...)"],
  "compatibility": "swi-9-compatible|needs-update|legacy",
  "complexity": "beginner|intermediate|advanced"
}

### ПРИМЕР:

#### Вход:
```prolog
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
"task": "Моделирование семейных отношений и родословных связей. Код определяет факты о прямых родительских связях между людьми (атомы-имена). Реализован рекурсивный поиск предков через цепочку родителей, сбор всех потомков конкретного человека в список с помощью findall/3, и подсчёт количества поколений (расстояния) между двумя людьми через арифметические вычисления.",
"domain": "family_relations",
"imports": ["library(lists)"],
"compatibility": "swi-9-compatible",
"complexity": "intermediate"
}}

Файл который необходимо проанализировать:
{content_preview}
    """

    model_name = os.getenv("OLLAMA_MODEL", "llama3")  # По умолчанию llama3

    for attempt in range(max_retries):
        try:
            # Создаем временный файл с промптом
            url = "http://localhost:11434/api/generate"
            
            payload = {
                "model": model_name,
                "prompt": prompt,
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

def process_prolog_file(file_path: Path, repo_path: str, repo_name: str) -> Tuple[bool, Dict]:
    try:
        # Пропускаем большие файлы
        
            
        # Читаем содержимое файла
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Проверяем сложность
        if not is_complex_prolog(content):
            return False, {}
        
        print(f"   🤖 Аннотируем с контекстом: {file_path.name}")
        
        # Собираем контекст всего репозитория
        repo_context = {}
        
        # 1. Ищем README в корне репозитория
        readme_content = ""
        for readme_name in ['README.md', 'README.txt', 'README', 'readme.md']:
            readme_path = Path(repo_path) / readme_name
            if readme_path.exists() and readme_path.is_file():
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        readme_content = f.read(1500)
                    repo_context['README'] = readme_content + "..." if len(readme_content) == 1500 else readme_content
                    break
                except:
                    continue
        
        
       
        
        
        # 4. Создаем промпт с полным контекстом
        context_str = "КОНТЕКСТ РЕПОЗИТОРИЯ:\n\n"
        
        if 'README' in repo_context:
            context_str += "=== README ===\n" + repo_context['README'] + "\n\n"
        
        
        
        
        prompt = f"""{context_str}

Проанализируй этот Prolog-код в контексте всего репозитория. Учти README, другие файлы и структуру проекта.

АНАЛИЗ ДОЛЖЕН ВКЛЮЧАТЬ:
1. Общее назначение файла в рамках проекта
2. Связи с другими файлами репозитория
3. Основные предикаты и их роль в системе
4. Архитектурные паттерны и логические конструкции
5. Как файл вписывается в общую архитектуру проекта

Ответ: кратко и информативно, 200-250 слов, на русском языке."""

        # 5. Выполняем аннотацию через Ollama
        annotation = ""
        
        for attempt in range(3):
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model":  os.getenv("OLLAMA_MODEL"),
                            "prompt": prompt,
                            "stream": False,
                            
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        annotation = result.get('response', '').strip()
                        break
                    else:
                        print(f"   ⚠️ Ошибка LLM (попытка {attempt + 1}): {response.status_code}")
                        time.sleep(3)
                        
                except requests.exceptions.RequestException as e:
                    print(f"   ⚠️ Ошибка подключения (попытка {attempt + 1}): {e}")
                    time.sleep(3)
                except Exception as e:
                    print(f"   ⚠️ Ошибка аннотации (попытка {attempt + 1}): {e}")
                    time.sleep(3)
        
        
        # 6. Формируем информацию о файле
        file_info = {
            'path': str(file_path),
            'filename': file_path.name,
            'annotation': annotation if annotation else "Аннотация не выполнена",
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