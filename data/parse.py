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
from prolog_analyze import is_complex_prolog, process_prolog_file, find_prolog_files
# Глобальная блокировка для безопасного доступа к файлам
file_lock = threading.Lock()





def analyze_and_annotate_repo(repo_path: str, repo_name: str) -> List[Dict]:
    """
    Анализирует репозиторий и создает аннотации с контекстом.
    """
    print(f"🔍 Анализ с контекстом: {repo_name}")
    
    prolog_files = find_prolog_files(repo_path)
    
    if not prolog_files:
        print(f"   ⚪ Нет Prolog-файлов в {repo_name}")
        return []
    
    print(f"   📄 Найдено Prolog-файлов: {len(prolog_files)}")
    
    complex_files = []
    
    # Проверяем каждый файл на сложность и аннотируем
    for file_path in prolog_files:
        is_complex, file_info = process_prolog_file(file_path, repo_path, repo_name)
        if is_complex:
            save_annotated_file(file_info=file_info, repo_info={'full_name': repo_name}, output_dir="data/annotated_repos")
            status = "✅" if file_info['annotation'] is not None else "⚠️"
            print(f"   {status} Обработан: {file_info['filename']}")
    
    print(f"   📊 Обработано сложных файлов: {len(complex_files)}")
    return complex_files

def remove_prolog_comments(code: str) -> str:
    """
    Удаляет Prolog-комментарии из кода:
    - Строчные: % ...
    - Блочные: /* ... */
    
    Сохраняет % внутри строк/атомов (например, '50%').
    """
    lines = code.split('\n')
    cleaned_lines = []
    in_block_comment = False
    
    for line in lines:
        cleaned_line = ""
        i = 0
        
        while i < len(line):
            # Проверка на блочный комментарий
            if in_block_comment:
                if i < len(line) - 1 and line[i:i+2] == '*/':
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue
            
            # Проверка на начало блочного комментария
            if i < len(line) - 1 and line[i:i+2] == '/*':
                in_block_comment = True
                i += 2
                continue
            
            # Проверка на строчный комментарий
            # Убедимся, что % не внутри строки (простая эвристика)
            if line[i] == '%':
                break  # Игнорируем всё до конца строки
            
            # Проверка на строки/атомы в кавычках (чтобы не удалять % внутри '50%')
            if line[i] in '"\'':
                quote = line[i]
                cleaned_line += line[i]
                i += 1
                while i < len(line):
                    if line[i] == '\\' and i + 1 < len(line):  # Экранирование
                        cleaned_line += line[i:i+2]
                        i += 2
                    elif line[i] == quote:
                        cleaned_line += line[i]
                        i += 1
                        break
                    else:
                        cleaned_line += line[i]
                        i += 1
                continue
            
            cleaned_line += line[i]
            i += 1
        
        # Убираем trailing пробелы после очистки
        cleaned_lines.append(cleaned_line.rstrip())
    
    # Убираем множественные пустые строки
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result

def save_annotated_file(file_info: Dict, repo_info: Dict, output_dir: str, 
                        remove_comments: bool = True) -> str:
    """
    Сохраняет аннотированный файл и его аннотацию в разные файлы.
    
    Args:
        file_info: Информация о файле (путь, аннотация, имя)
        repo_info: Информация о репозитории
        output_dir: Корневая папка для сохранения
        remove_comments: Если True, удаляет Prolog-комментарии из кода
    
    Returns:
        Путь к папке репозитория или пустая строла при ошибке
    """
    try:
        # 1. Создаем папку для репозитория
        repo_name = repo_info['full_name'].replace('/', '_')
        repo_output_dir = os.path.join(output_dir, repo_name)
        os.makedirs(repo_output_dir, exist_ok=True)
        
        # 2. Получаем базовое имя файла без расширения
        original_filename = file_info['filename']
        base_name = Path(original_filename).stem  # Убираем .pl
        
        # 3. Читаем исходный код
        src_path = file_info['path']
        with open(src_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 4. Очищаем от комментариев если нужно
        if remove_comments:
            code = remove_prolog_comments(code)
            print(f"   🧹 Комментарии удалены из {base_name}.pl")
        
        # 5. Сохраняем очищенный/оригинальный .pl файл
        dst_pl_path = os.path.join(repo_output_dir, f"{base_name}.pl")
        with open(dst_pl_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 6. Сохраняем аннотацию в .txt
        annotation_txt_path = os.path.join(repo_output_dir, f"{base_name}.txt")
        with open(annotation_txt_path, 'w', encoding='utf-8') as f:
            f.write(file_info.get('annotation', ''))
        
        print(f"   💾 Сохранены: {base_name}.pl и {base_name}.txt")
        return repo_output_dir
        
    except Exception as e:
        print(f"⚠️ Ошибка сохранения {file_info.get('filename', 'unknown')}: {e}")
        return ""
def process_repos(
    repos: List[Dict],
    download_dir: str = "data/repos",
    output_dir: str = "data/annotated_repos",
    delay_between: float = 20.0
) -> List[Dict]:
    """
    Скачивает, анализирует, аннотирует и сохраняет репозитории.
    
    Returns:
        List[Dict]: Список информации о сохранённых репозиториях
    """
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for i, repo in enumerate(repos, 1):
        repo_name = repo['full_name']
        print(f"\n[{i}/{len(repos)}] Обработка: {repo_name}")
        print("=" * 60)
        
        # 1. Скачиваем репозиторий
        downloaded_path = download_github_repo(repo_name, output_dir=download_dir)
        if downloaded_path is None:
            print(f"⏭️ Пропускаем {repo_name} — не удалось скачать")
            time.sleep(delay_between)
            continue
        
        # 2. Анализируем и аннотируем файлы
        annotated_files = analyze_and_annotate_repo(downloaded_path, repo_name)
        
        
        # 4. Удаляем исходный репозиторий
        try:
            shutil.rmtree(downloaded_path)
            print(f"🗑️ Удалён исходный: {downloaded_path}")
        except Exception as e:
            print(f"⚠️ Не удалось удалить {downloaded_path}: {e}")
        
        # Задержка между репозиториями
        if i < len(repos):
            print(f"😴 Ждём {delay_between}с перед следующим...")
            time.sleep(delay_between)
    
    # Итоговая статистика
    print(f"\n🎯 Итоги:")
    print(f"   Всего обработано: {len(repos)}")
    print(f"   Сохранено с аннотациями: {len(results)}")
    total_annotated = sum(r['annotated_files_count'] for r in results)
    print(f"   Всего аннотированных файлов: {total_annotated}")
    print(f"   Папка с результатами: {os.path.abspath(output_dir)}")
    
    return results

# ===== ЗАПУСК =====
if __name__ == "__main__":
    # Загружаем переменные окружения из .env
    from dotenv import load_dotenv
    load_dotenv()
    
    print("🚀 Запуск анализа и аннотации Prolog-репозиториев")
    print(f"📁 Папка для скачивания: repos/")
    print(f"📁 Папка для аннотаций: annotated_repos/")
    print(f"🤖 Используем Ollama для аннотаций")
    print("-" * 60)
    
    ranges = list(zip(range(5,200,25),range(30,225,25)))
    ranges.reverse()
    d = []
    test_value = 0
    d = get_prolog_repos(max_pages=1000, per_page=10,min_stars=225)
    for s,e in ranges:
        if len(d)>test_value:
            break
        d.extend(get_prolog_repos(max_pages=1000, per_page=10, min_stars=s, max_stars=e))
        

      # Уменьшаем для тестирования
    d.reverse()
    print(f"📋 Найдено репозиториев: {len(d)}")
    
    if not d:
        print("❌ Нет репозиториев для обработки")
    else:
        # Обрабатываем
        annotated_repos = process_repos(d)
        
        # Сохраняем общий отчет
        if annotated_repos:
            report = {
                'total_repos_processed': len(d),
                'annotated_repos_count': len(annotated_repos),
                'total_annotated_files': sum(r['annotated_files_count'] for r in annotated_repos),
                'annotated_repos': annotated_repos,
                'processed_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open("annotation_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"📄 Отчёт сохранён: annotation_report.json")