import os
import time
from math import ceil
from math import log10
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - optional during offline tests
    requests = None
from data.pipeline_constants import (
    GITHUB_MAX_REPOS,
    GITHUB_MAX_STARS,
    GITHUB_MIN_STARS,
    GITHUB_MIN_STAR_BUCKET_SIZE,
    GITHUB_PAGE_DELAY_SEC,
    GITHUB_RATE_LIMIT_DELAY_SEC,
    GITHUB_SEARCH_RESULT_LIMIT,
    GITHUB_SEARCH_TIMEOUT_SEC,
    GITHUB_SEARCH_URL,
)


def _is_primary_prolog_repo(item: dict[str, Any]) -> bool:
    return item.get("language") == "Prolog"


def _get_star_bucket_size(bucket_max: int) -> int:
    if bucket_max <= 0:
        return GITHUB_MIN_STAR_BUCKET_SIZE

    magnitude_step = 10 ** max(0, int(log10(bucket_max)) - 1)
    return max(GITHUB_MIN_STAR_BUCKET_SIZE, magnitude_step)


def _build_headers() -> dict[str, str]:
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _build_query(min_stars: int, max_stars: int | None) -> str:
    if max_stars is None:
        return f"language:Prolog stars:>={min_stars}"
    return f"language:Prolog stars:{min_stars}..{max_stars}"


def _request_search_page(
    headers: dict[str, str],
    min_stars: int,
    max_stars: int | None,
    page: int,
    per_page: int,
    max_retries: int,
    retry_delay: int,
) -> dict[str, Any] | None:
    if requests is None:
        raise RuntimeError("requests is required to query the GitHub Search API")

    params = {
        "q": _build_query(min_stars, max_stars),
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
        "page": page,
    }

    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"Query stars={min_stars}..{max_stars or 'inf'}, "
                f"page {page}, attempt {attempt}/{max_retries}"
            )
            response = requests.get(
                GITHUB_SEARCH_URL,
                headers=headers,
                params=params,
                timeout=GITHUB_SEARCH_TIMEOUT_SEC,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code == 403 and "rate limit" in response.text.lower():
                print(f"Rate limit exceeded. Waiting {GITHUB_RATE_LIMIT_DELAY_SEC} seconds...")
                time.sleep(GITHUB_RATE_LIMIT_DELAY_SEC)
                continue

            print(f"GitHub API error {response.status_code}: {response.text}")
            return None
        except requests.exceptions.ConnectionError as exc:
            print(f"Connection error (attempt {attempt}): {exc}")
        except requests.exceptions.Timeout as exc:
            print(f"Request timeout (attempt {attempt}): {exc}")
        except requests.exceptions.RequestException as exc:
            print(f"Request error (attempt {attempt}): {exc}")

        if attempt < max_retries:
            print(f"Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)

    print(f"Failed to fetch stars={min_stars}..{max_stars or 'inf'}, page {page}")
    return None


def _discover_max_stars(
    headers: dict[str, str],
    min_stars: int,
    max_pages: int,
    per_page: int,
    max_retries: int,
    retry_delay: int,
) -> int | None:
    page = 1

    while max_pages <= 0 or page <= max_pages:
        data = _request_search_page(
            headers=headers,
            min_stars=min_stars,
            max_stars=None,
            page=page,
            per_page=per_page,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        if not data:
            return None

        items = data.get("items", [])
        if not items:
            return None

        for item in items:
            if _is_primary_prolog_repo(item):
                stars = item.get("stargazers_count", min_stars)
                print(
                    "Discovered upper star bound "
                    f"{stars} from {item.get('full_name', '<unknown repo>')}"
                )
                return stars

        if len(items) < per_page:
            return None

        page += 1
        time.sleep(GITHUB_PAGE_DELAY_SEC)

    return None


def _fetch_star_bucket(
    headers: dict[str, str],
    min_stars: int,
    max_stars: int,
    max_repos: int,
    max_pages: int,
    per_page: int,
    max_retries: int,
    retry_delay: int,
    seen_full_names: set[str],
) -> list[dict[str, Any]]:
    repos: list[dict[str, Any]] = []
    page = 1
    max_bucket_pages = ceil(GITHUB_SEARCH_RESULT_LIMIT / per_page)

    while max_pages <= 0 or page <= max_pages:
        if len(seen_full_names) >= max_repos:
            print(f"Reached repository limit: {max_repos}")
            break

        if page > max_bucket_pages:
            print(
                f"Bucket {min_stars}..{max_stars} hit the GitHub Search limit "
                f"of {GITHUB_SEARCH_RESULT_LIMIT} accessible results."
            )
            break

        data = _request_search_page(
            headers=headers,
            min_stars=min_stars,
            max_stars=max_stars,
            page=page,
            per_page=per_page,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        if not data:
            break

        items = data.get("items", [])
        if not items:
            break

        kept = 0
        skipped_non_prolog = 0
        for item in items:
            if len(seen_full_names) >= max_repos:
                break

            if not _is_primary_prolog_repo(item):
                skipped_non_prolog += 1
                continue

            full_name = item.get("full_name")
            if not full_name or full_name in seen_full_names:
                continue
            seen_full_names.add(full_name)
            repos.append(item)
            kept += 1

        print(
            f"Bucket {min_stars}..{max_stars}: received {len(items)} "
            f"items on page {page}, kept {kept}, skipped_non_prolog {skipped_non_prolog}"
        )

        if len(items) < per_page:
            break

        page += 1
        time.sleep(GITHUB_PAGE_DELAY_SEC)

    return repos


def get_prolog_repos(
    min_stars: int = GITHUB_MIN_STARS,
    max_stars: int | None = GITHUB_MAX_STARS or None,
    max_repos: int = GITHUB_MAX_REPOS,
    max_pages: int = 100,
    per_page: int = 30,
    max_retries: int = 3,
    retry_delay: int = 40,
) -> list[dict[str, Any]]:
    """
    Fetch GitHub repositories where the main language is Prolog.

    GitHub Search is queried in fixed star buckets to avoid relying on a
    single capped result set.
    """
    headers = _build_headers()
    upper_bound = max_stars
    if upper_bound is None:
        upper_bound = _discover_max_stars(
            headers=headers,
            min_stars=min_stars,
            max_pages=max_pages,
            per_page=per_page,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        if upper_bound is None:
            print("Unable to discover the upper star bound for Prolog repositories.")
            return []

    repos: list[dict[str, Any]] = []
    seen_full_names: set[str] = set()
    bucket_max = upper_bound

    while bucket_max >= min_stars and len(repos) < max_repos:
        bucket_size = _get_star_bucket_size(bucket_max)
        bucket_min = max(min_stars, bucket_max - bucket_size + 1)
        print(f"Scanning star bucket {bucket_min}..{bucket_max} (step={bucket_size})")
        repos.extend(
            _fetch_star_bucket(
                headers=headers,
                min_stars=bucket_min,
                max_stars=bucket_max,
                max_repos=max_repos,
                max_pages=max_pages,
                per_page=per_page,
                max_retries=max_retries,
                retry_delay=retry_delay,
                seen_full_names=seen_full_names,
            )
        )
        bucket_max = bucket_min - 1

    if len(repos) >= max_repos:
        print(f"Reached repository limit: {max_repos}")

    print(f"Total repositories fetched: {len(repos)}")
    return repos
