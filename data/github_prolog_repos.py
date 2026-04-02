import requests
import time
from typing import List, Dict
import os

def get_prolog_repos(
    min_stars: int = 5,
    max_stars: int = -1, 
    max_pages: int = 1000, 
    per_page: int = 30, 
    start_page: int = 0,
    max_retries: int = 3,
    retry_delay: int = 40
) -> List[Dict]:
    """
    Получает список репозиториев с GitHub, где основной язык — Prolog.
    
    Args:
        min_stars: Минимальное количество звёзд
        max_pages: Максимальное количество страниц для обхода
        per_page: Количество результатов на странице (макс. 100)
        start_page: Стартовая страница
        max_retries: Максимальное число попыток при ошибке соединения
        retry_delay: Задержка между попытками в секундах (по умолчанию 40)
    
    Returns:
        List[Dict]: Список словарей с информацией о репозиториях
    """
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    repos = []
    end_of_pages = False
    for page in range(start_page, max_pages + 1):
        if max_stars>0:
            url = (
                f"https://api.github.com/search/repositories?"
                f"q=language:Prolog+stars:>={min_stars}+stars:<={max_stars}" 
                f"&sort=stars&order=desc&per_page={per_page}&page={page}"
            )
        else:
            url = (
                f"https://api.github.com/search/repositories?"
                f"q=language:Prolog+stars:>={min_stars}" 
                f"&sort=stars&order=desc&per_page={per_page}&page={page}"
            )
        
        # ===== Логика повторных попыток при разрыве соединения =====
        for attempt in range(max_retries):
            try:
                print(f"📄 Страница {page}, попытка {attempt + 1}/{max_retries}...")
                response = requests.get(url, headers=headers, timeout=30)
                
                # Обработка успешного ответа
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])
                    repos.extend(items)
                    print(f"✅ Получено {len(items)} репозиториев со страницы {page}")
                    
                    # Если на странице меньше элементов, чем per_page — это последняя
                    if len(items) < per_page:
                        print("Достигнута последняя страница")
                        end_of_pages = True
                        break
                    
                    # Rate limiting для успешных запросов
                    time.sleep(20)
                    break  # Выход из цикла повторных попыток
                
                # Обработка ошибок API
                elif response.status_code == 403 and "rate limit" in response.text.lower():
                    print(f"⏱️ Превышен лимит запросов. Ждём 60 секунд...")
                    time.sleep(60)
                    continue  # Повторить текущую страницу
                
                elif response.status_code >= 400:
                    print(f"❌ Ошибка API {response.status_code}: {response.text}")
                    end_of_pages = True
                    break  # Не повторяем клиентские ошибки
                
                # Сетевые ошибки и таймауты
            except requests.exceptions.ConnectionError as e:
                print(f"🔌 Разрыв соединения (попытка {attempt + 1}): {e}")
            except requests.exceptions.Timeout as e:
                print(f"⏰ Таймаут запроса (попытка {attempt + 1}): {e}")
            except requests.exceptions.RequestException as e:
                print(f"🌐 Ошибка запроса (попытка {attempt + 1}): {e}")
            
            # ===== Задержка перед следующей попыткой =====
            if attempt < max_retries - 1:
                print(f"😴 Ждём {retry_delay} секунд перед повторной попыткой...")
                time.sleep(retry_delay)
            else:
                print(f"❌ Не удалось получить страницу {page} после {max_retries} попыток")
                break  # Переходим к следующей странице или завершаем
                
        if end_of_pages:
                break
    print(f"🎯 Всего получено репозиториев: {len(repos)}")
    return repos
def get_repo_urls(repos: List[Dict]) -> List[str]:
    """
    Возвращает список URL-ов репозиториев из результата get_prolog_repos.
    """
    return [repo["html_url"] for repo in repos]

