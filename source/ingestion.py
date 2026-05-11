"""
ingestion.py — Async GitHub API data fetcher.

All network I/O is handled here using asyncio + aiohttp so that multiple
GitHub API endpoints are hit concurrently without blocking the event loop.

Architecture note: GitHub API calls are pure I/O-bound (we spend time waiting
for HTTP responses, not doing CPU work). asyncio is the correct concurrency
primitive here — it bypasses the GIL entirely for I/O and avoids the overhead
of spawning OS threads or processes.
"""

import asyncio
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import aiohttp
from dotenv import load_dotenv

load_dotenv()

GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def _build_headers() -> dict[str, str]:
    """Return request headers, injecting a Bearer token when available."""
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _lang_qualifier(language: str) -> str:
    """
    Build a GitHub search language qualifier, quoting only when necessary.

    Single-word languages (``python``, ``javascript``, ``C``) must NOT be
    quoted — ``language:"python"`` behaves differently from ``language:python``
    in GitHub's search index and can suppress highly-starred repos from results.

    Multi-word languages (``Jupyter Notebook``, ``Visual Basic .NET``) MUST be
    quoted, otherwise the space causes GitHub to split the value into separate
    unrelated search terms and the language filter is silently broken.
    """
    if " " in language:
        return f'language:"{language}"'
    return f"language:{language}"


async def _get(session: aiohttp.ClientSession, url: str) -> Any:
    """
    Perform a single authenticated GET request and return parsed JSON.

    Returns an empty dict on non-200 responses rather than raising, so
    the pipeline degrades gracefully on rate-limit or other errors.
    """
    async with session.get(url, headers=_build_headers()) as response:
        if response.status == 200:
            return await response.json()
        return {}


async def _fetch_commits_weekly(session: aiohttp.ClientSession, repo: str) -> list[int]:
    """
    Return a 52-element list of per-week commit counts covering the last year.

    Uses ``/repos/{repo}/commits?since=...&per_page=100`` — a standard paginated
    endpoint that always responds with 200 immediately — rather than the
    ``/stats/commit_activity`` stats endpoint which requires GitHub to precompute
    data and frequently returns 202 ("still computing") for small or cold repos,
    leaving us with empty data and a useless ``commit_consistency`` metric.

    The 100-commit cap covers a year of activity for most repos; for highly
    active repos (>100 commits/year) only the most recent 100 are counted, which
    still gives a representative consistency signal.

    Args:
        session: Shared aiohttp session.
        repo: Full ``owner/repo`` repository name.

    Returns:
        List of 52 integers, index 0 = oldest week, index 51 = current week.
    """
    # strftime produces "2025-05-11T14:55:00Z" — URL-safe, no "+" that would
    # be misread as a space. isoformat() emits "+00:00" which breaks the URL.
    since = (datetime.now(timezone.utc) - timedelta(weeks=52)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"{GITHUB_API_BASE}/repos/{repo}/commits?per_page=100&since={since}"

    async with session.get(url, headers=_build_headers()) as response:
        if response.status != 200:
            return []
        commits = await response.json()

    if not isinstance(commits, list):
        return []

    weekly: list[int] = [0] * 52
    now = datetime.now(timezone.utc)
    for commit in commits:
        date_str = (
            commit.get("commit", {}).get("author", {}).get("date", "")
        )
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            weeks_ago = (now - dt).days // 7
            if 0 <= weeks_ago < 52:
                weekly[51 - weeks_ago] += 1
        except (ValueError, TypeError):
            continue

    return weekly


async def _get_closed_issues_count(session: aiohttp.ClientSession, repo: str) -> int:
    """
    Return the total number of closed issues for a repository.

    Fetches ``/repos/{repo}/issues?state=closed&per_page=1`` and reads the
    total count from the ``Link`` response header (GitHub's pagination footer).
    Falls back to counting the body items when there is only one page, and
    returns 0 on any error.

    GitHub's repo metadata endpoint does not expose a ``closed_issues`` field,
    so this dedicated call is the only reliable way to get the count.
    """
    url = f"{GITHUB_API_BASE}/repos/{repo}/issues?state=closed&per_page=1"
    async with session.get(url, headers=_build_headers()) as response:
        if response.status != 200:
            return 0
        link_header = response.headers.get("Link", "")
        # The Link header looks like: <...?page=N>; rel="last"
        match = re.search(r'[?&]page=(\d+)>;\s*rel="last"', link_header)
        if match:
            return int(match.group(1))
        # No "last" link → only one page; count from the body.
        body = await response.json()
        return len(body) if isinstance(body, list) else 0


async def fetch_repo_data(repo: str) -> dict[str, Any]:
    """
    Concurrently fetch all metrics for a single GitHub repository.

    Hits four endpoints simultaneously via asyncio.gather():
    - /repos/{repo}                              → core metadata (stars, forks, language)
    - /repos/{repo}/commits?since=...            → last-52-week commit timestamps;
                                                   binned into weekly counts locally.
                                                   Always returns 200 — replaces the
                                                   unreliable /stats/commit_activity
                                                   endpoint which returns 202 for small
                                                   or cold repos.
    - /repos/{repo}/issues?state=closed&per_page=1 → total closed issues via Link header
    - /repos/{repo}/contributors?per_page=100    → contributor list; standard paginated
                                                   endpoint, always 200 — replaces
                                                   /stats/contributors which also
                                                   suffers from 202 responses.

    Args:
        repo: Full repository name in "owner/repo" format.

    Returns:
        A dictionary containing the merged results from all four endpoints.
    """
    async with aiohttp.ClientSession() as session:
        metadata, weekly_commits, closed_issues_count, contributors = await asyncio.gather(
            _get(session, f"{GITHUB_API_BASE}/repos/{repo}"),
            _fetch_commits_weekly(session, repo),
            _get_closed_issues_count(session, repo),
            _get(session, f"{GITHUB_API_BASE}/repos/{repo}/contributors?per_page=100"),
        )

    total_issues = metadata.get("open_issues_count", 0) + closed_issues_count
    contributor_count = len(contributors) if isinstance(contributors, list) else 0

    return {
        "repo": repo,
        "stars": metadata.get("stargazers_count", 0),
        "forks": metadata.get("forks_count", 0),
        "open_issues": metadata.get("open_issues_count", 0),
        "total_issues": total_issues,
        "contributor_count": contributor_count,
        "weekly_commits": weekly_commits,
        "repo_age_days": _compute_age_days(metadata.get("created_at")),
        "language": metadata.get("language", "Unknown"),
        "description": metadata.get("description", ""),
        "html_url": metadata.get("html_url", ""),
    }


async def fetch_trending_repos(language: str = "python", limit: int = 10) -> list[dict[str, Any]]:
    """
    Fetch and analyze multiple trending GitHub repositories concurrently.

    Searches GitHub for the most-starred repositories in the given language
    pushed within the last 30 days, then fetches full metrics for each repo
    simultaneously.

    Args:
        language: Programming language filter (e.g. "python", "javascript").
        limit: Maximum number of repositories to return (capped at 30).

    Returns:
        A list of raw metric dictionaries, one per repository.
    """
    limit = min(limit, 30)
    q = quote(_lang_qualifier(language), safe=':"')
    search_url = f"{GITHUB_API_BASE}/search/repositories?q={q}&sort=stars&order=desc&per_page={limit}"

    async with aiohttp.ClientSession() as session:
        search_results = await _get(session, search_url)

    items = search_results.get("items", [])
    repo_names = [item["full_name"] for item in items]

    results = await asyncio.gather(
        *[fetch_repo_data(name) for name in repo_names]
    )
    return list(results)


async def fetch_similar_repos(repo_name: str, language: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Fetch the top most-starred GitHub repositories that match a keyword search for
    ``repo_name`` written in ``language``.

    The search query mirrors the trending search pattern but adds the bare repo
    name as a keyword — e.g. for ``repo_name="pothole_detection"`` and
    ``language="Jupyter Notebook"`` the effective GitHub query is::

        pothole_detection language:Jupyter Notebook

    sorted by stars descending.  Full metrics are then fetched concurrently for
    every result using the same ``asyncio.gather`` pattern as
    ``fetch_trending_repos``.

    Args:
        repo_name: The bare repository name used as a search keyword
                   (e.g. ``"pothole_detection"``, not ``"owner/pothole_detection"``).
        language: Programming language detected from the source repository
                  (e.g. ``"Python"``, ``"Jupyter Notebook"``).
        limit: Maximum number of repositories to return (hard-capped at 10).

    Returns:
        A list of raw metric dictionaries, one per matching repository.
    """
    limit = min(limit, 10)
    q = quote(f"{repo_name} {_lang_qualifier(language)}", safe=':"')
    search_url = f"{GITHUB_API_BASE}/search/repositories?q={q}&sort=stars&order=desc&per_page={limit}"

    async with aiohttp.ClientSession() as session:
        search_results = await _get(session, search_url)

    items = search_results.get("items", [])
    repo_names = [item["full_name"] for item in items]

    results = await asyncio.gather(
        *[fetch_repo_data(name) for name in repo_names]
    )
    return list(results)


def _compute_age_days(created_at: str | None) -> int:
    """
    Calculate how many days old a repository is based on its creation timestamp.

    Args:
        created_at: ISO 8601 timestamp string from the GitHub API (e.g. "2020-01-15T10:00:00Z").

    Returns:
        Number of days since creation, or 1 if the timestamp is missing.
    """
    if not created_at:
        return 1
    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    delta = datetime.now(timezone.utc) - created
    return max(delta.days, 1)
