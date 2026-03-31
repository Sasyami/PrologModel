import os
import requests
import zipfile
from pathlib import Path
from typing import Optional
import shutil
import time

def download_github_repo(
    repo_url: str, 
    output_dir: str = "data",
    branch: str = None,
    extract: bool = True,
    remove_zip: bool = True,
    max_retries: int = 3  # Добавляем параметр для количества попыток
) -> Optional[str]:
    """
    Скачивает репозиторий с GitHub в указанную папку.
    
    Args:
        repo_url (str): URL репозитория GitHub
        output_dir (str): Папка для сохранения
        branch (str): Ветка для скачивания
        extract (bool): Распаковать архив
        remove_zip (bool): Удалить zip файл после распаковки
        max_retries (int): Количество попыток скачивания
    """
    if "github.com" in repo_url:
                # Извлекаем username/repo из полного URL
                parts = repo_url.rstrip('/').split('/')
                if len(parts) >= 2:
                    username = parts[-2]
                    repo_name = parts[-1]
                    repo_identifier = f"{username}/{repo_name}"
                else:
                    raise ValueError("Неверный формат URL GitHub")
    else:
                # Формат "username/repo"
                if '/' in repo_url:
                    username, repo_name = repo_url.split('/', 1)
                    repo_identifier = repo_url
                else:
                    raise ValueError("Неверный формат. Ожидается 'username/repo' или полный URL")
    token = os.getenv("GITHUB_TOKEN")
    headers = {
                        "Authorization": f"token {token}",
                        "Accept": "application/vnd.github.v3+json"
                    }
    branch_to_use = ['main', 'master'] if branch is None else [branch]
    for b in branch_to_use:
                url = f"https://github.com/{repo_identifier}/archive/refs/heads/{b}.zip"
                try:
                    # Используем HEAD, чтобы не скачивать файл целиком
                    r = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
                    if r.status_code == 200:
                        print(f"✅ Найдена ветка: {b}")
                        branch = b
                except:
                    continue
    for attempt in range(max_retries):
        try:
            # Сначала извлекаем username и repo_name
            
            
            # Обработка разных форматов URL
           
            
            
            
            # Если ничего не найдено, возвращаем main по умолчанию
            branch = branch or 'main'
                    

            time.sleep(5)  # Небольшая задержка перед скачиванием
            # Формируем URL для скачивания
            download_url = f"https://github.com/{repo_identifier}/archive/refs/heads/{branch}.zip"
            
            # Создаем папку для сохранения
            os.makedirs(output_dir, exist_ok=True)
            
            # Имя файла для сохранения
            zip_filename = os.path.join(output_dir, f"{repo_identifier.replace('/', '_')}_{branch}.zip")
            
            print(f"Скачивание репозитория {repo_identifier} (ветка: {branch})...")
            if attempt > 0:
                print(f"Попытка {attempt + 1}/{max_retries}")
            if token:
                    headers = {
                        "Authorization": f"token {token}",
                        "Accept": "application/vnd.github.v3+json"
                    }
            # Скачиваем файл с таймаутом и повторными попытками
            response = requests.get(download_url,headers=headers, stream=True, timeout=10)
            response.raise_for_status()
            
            with open(zip_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Проверяем, что chunk не пустой
                        f.write(chunk)
            
            print(f"Репо сохранен: {zip_filename}")
            
            if extract:
                # Распаковываем архив
                extract_path = os.path.join(output_dir, f"{repo_identifier.replace('/', '_')}_{branch}")
                
                with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                
                # Ищем распакованную папку
                extracted_folders = []
                if repo_name:
                    extracted_folders = [f for f in os.listdir(output_dir) 
                                       if f.startswith(f"{repo_name}-") and os.path.isdir(os.path.join(output_dir, f))]
                
                if extracted_folders:
                    # Используем папку как есть, без переименования
                    actual_path = os.path.join(output_dir, extracted_folders[0])
                    print(f"Репо распакован: {actual_path}")
                    
                    if remove_zip:
                        try:
                            os.remove(zip_filename)
                            print("ZIP файл удален")
                        except:
                            print("Не удалось удалить ZIP файл")
                    
                    return actual_path
                else:
                    print("⚠️ Не найдена распакованная папка")
                    # Удаляем битый zip файл
                    try:
                        os.remove(zip_filename)
                    except:
                        pass
                    return None
            else:
                return zip_filename
            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при скачивании (попытка {attempt + 1}/{max_retries}): {e}")
            
            # Удаляем частично скачанный файл если он существует
            if 'zip_filename' in locals() and os.path.exists(zip_filename):
                try:
                    os.remove(zip_filename)
                except:
                    pass
            
            if attempt < max_retries - 1:
                wait_time = 30  # Ждем 30 секунд перед следующей попыткой
                print(f"Ждем {wait_time} секунд перед повторной попыткой...")
                time.sleep(wait_time)
                continue
            else:
                return None
                
        except zipfile.BadZipFile:
            print(f"Ошибка: битый ZIP файл (попытка {attempt + 1}/{max_retries})")
            # Удаляем битый zip файл
            if 'zip_filename' in locals() and os.path.exists(zip_filename):
                try:
                    os.remove(zip_filename)
                except:
                    pass
            
            if attempt < max_retries - 1:
                wait_time = 30 # Ждем 30 секунд перед следующей попыткой
                print(f"Ждем {wait_time} секунд перед повторной попыткой...")
                time.sleep(wait_time)
                continue
            else:
                return None
                
        except Exception as e:
            print(f"Произошла ошибка (попытка {attempt + 1}/{max_retries}): {e}")
            
            # Удаляем частично скачанный файл если он существует
            if 'zip_filename' in locals() and os.path.exists(zip_filename):
                try:
                    os.remove(zip_filename)
                except:
                    pass
            
            if attempt < max_retries - 1:
                wait_time = 30 # Ждем 30 секунд перед следующей попыткой
                print(f"Ждем {wait_time} секунд перед повторной попыткой...")
                time.sleep(wait_time)
                continue
            else:
                return None
    
    return None

