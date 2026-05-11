"""
api.py — FastAPI application and route handlers.

Exposes three endpoints:
  GET /health
      Liveness check.

  GET /api/analyze?repo=owner/repo
      Fetches full metrics for the given repository, then searches GitHub for
      the top 10 most-starred repos sharing the same name and primary language.
      Batch z-score statistical analysis runs only when at least 2 similar
      repositories are found; otherwise only repo metadata is returned with a
      note explaining why analysis was skipped.

  GET /api/trending?language=python&limit=10
      Fetches the top starred repositories for a language and scores all of
      them relative to each other via z-score normalization.

Each route is async end-to-end: the FastAPI handler awaits the ingestion
coroutines, then calls the (synchronous but fast) processor and analyzer.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from source.analyzer import score_repos
from source.ingestion import fetch_repo_data, fetch_similar_repos, fetch_trending_repos
from source.processor import prepare_pipeline, save_pipeline_data

app = FastAPI(
    title="GitHub Insights API",
    description=(
        "Scans GitHub repositories and answers: **is this software actually useful, "
        "or is it just getting empty attention?**\n\n"
        "### How it works\n"
        "1. `/api/analyze` fetches full metrics for a repository, detects its primary "
        "language, then searches GitHub for the top 10 most-starred repositories "
        "sharing the same name and language.\n"
        "   - If **2 or more** similar repos are found, all repos (including yours) "
        "are scored together using z-score normalisation so every score reflects "
        "where a repo sits relative to its real peers.\n"
        "   - If **fewer than 2** similar repos are found, statistical comparison is "
        "meaningless and is skipped — only raw metadata is returned with a note.\n"
        "2. `/api/trending` scores the top starred repos for a language against each other.\n\n"
        "### Scoring\n"
        "Each repository receives a **Hype Score** (0–10) and a **Substance Score** (0–10) "
        "computed from z-score-normalised metrics:\n"
        "- **Hype**: star velocity, raw star count\n"
        "- **Substance**: commit consistency, issue resolution rate, "
        "fork-substance ratio, contributor count\n\n"
        "Verdict is **Useful** (substance > 5), **Just Hype** (hype dominates), "
        "or **Emerging** (not enough signal yet)."
    ),
    version="1.1.0",
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect the browser root to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Meta"])
async def health_check() -> dict:
    """
    Liveness probe — confirms the service is running.

    Returns `{"status": "ok"}`. Use this for deployment health checks or a
    quick smoke test before making analysis requests.
    """
    return {"status": "ok", "service": "GitHub Insights API"}


@app.get("/api/analyze", tags=["Analysis"])
async def analyze_repo(
    repo: str = Query(
        ...,
        description=(
            'Full repository name in **"owner/repo"** format. '
            'Example: `fastapi/fastapi`, `aram199922/pothole_detection`.'
        ),
        examples=["fastapi/fastapi"],
    ),
) -> dict:
    """
    Analyze a repository and compare it against the top similar repos on GitHub.

    **Pipeline**

    1. Fetches repository metadata, last-year commit history, closed-issue count,
       and contributor list — all four requests fire concurrently.
    2. Detects the primary language from the metadata.
    3. Searches GitHub for `{repo_name} language:"{lang}"` sorted by stars
       (e.g. `pothole_detection language:"Jupyter Notebook"`).
    4. **If 2 or more similar repos are found**: combines the source repo with
       the similar repos into one batch, runs `prepare_pipeline` + `score_repos`
       so every score reflects real peer comparison via z-score normalisation.
       The processed DataFrame is saved to `_data/api_data/` as an Excel file
       for offline inspection.
    5. **If fewer than 2 similar repos are found**: statistical comparison is
       skipped — z-scores across 0 or 1 peers are meaningless. Raw metadata is
       returned with `scores: null` and a note in `similar_repos`.

    **Returns**

    - `repo`, `meta`, `raw_metrics` — always present.
    - `verdict`, `confidence`, `scores` — present only when batch analysis ran
      (i.e. at least 2 similar repos were found); `null` otherwise.
    - `similar_repos` — ranked batch results when analysis ran; a short note
      with the count when it was skipped; `null` when language is unknown.

    **Errors**

    - `400` — repo is not in `owner/repo` format.
    """
    if "/" not in repo or len(repo.split("/")) != 2:
        raise HTTPException(
            status_code=400,
            detail='Repository must be in "owner/repo" format (e.g. "fastapi/fastapi").',
        )

    raw = await fetch_repo_data(repo)

    language = raw.get("language") or ""
    repo_name = repo.split("/")[1]
    similar_section = None
    result = None

    if language and language != "Unknown":
        similar_raw = await fetch_similar_repos(repo_name=repo_name, language=language)
        count = len(similar_raw)

        if count < 2:
            # Not enough peers — z-score comparison would be meaningless.
            noun = "repository" if count == 1 else "repositories"
            similar_section = {
                "count": count,
                "note": (
                    f"Only {count} similar {noun} found. "
                    "Statistical comparison requires at least 2 similar repositories."
                ),
            }
        else:
            # Source repo is first so all_scored[0] is always the source.
            combined_raw = [raw] + similar_raw
            df = prepare_pipeline(combined_raw)
            save_pipeline_data(df, f"analyze_{repo.replace('/', '_')}")
            all_scored = score_repos(df)

            result = all_scored[0]
            peer_scored = all_scored[1:]
            ranked = sorted(peer_scored, key=lambda r: r.substance_score, reverse=True)

            similar_section = {
                "search_query": f'{repo_name} language:"{language}"',
                "count": len(ranked),
                "results": [
                    {
                        "rank": idx + 1,
                        "repo": r.repo,
                        "verdict": r.verdict,
                        "confidence": r.confidence,
                        "scores": {
                            "hype_score": r.hype_score,
                            "substance_score": r.substance_score,
                        },
                        "meta": {
                            "language": r.language,
                            "description": r.description,
                            "html_url": r.html_url,
                        },
                    }
                    for idx, r in enumerate(ranked)
                ],
            }

    return {
        "repo": repo,
        "verdict": result.verdict if result else None,
        "confidence": result.confidence if result else None,
        "scores": (
            {"hype_score": result.hype_score, "substance_score": result.substance_score}
            if result else None
        ),
        "meta": {
            "language": language or "Unknown",
            "description": raw.get("description", ""),
            "html_url": raw.get("html_url", ""),
        },
        "raw_metrics": {
            "stars": raw.get("stars"),
            "forks": raw.get("forks"),
            "open_issues": raw.get("open_issues"),
            "total_issues": raw.get("total_issues"),
            "contributor_count": raw.get("contributor_count"),
            "repo_age_days": raw.get("repo_age_days"),
        },
        "similar_repos": similar_section,
    }


@app.get("/api/trending", tags=["Analysis"])
async def trending_repos(
    language: str = Query(
        default="python",
        description=(
            "Programming language to search by. Single-word values like `python` "
            "or multi-word values like `Jupyter Notebook` are both supported."
        ),
        examples=["python"],
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=30,
        description="Number of repositories to fetch and score (1–30, default 10).",
    ),
) -> dict:
    """
    Score the top most-starred GitHub repositories for a given programming language.

    All repositories are scored **relative to each other** using z-score
    normalisation across the full batch, so a score of 7 means "notably above
    the average of this peer group", not an absolute quality judgment.

    **Pipeline**

    1. Searches GitHub for `language:"{lang}"` sorted by stars descending.
    2. Fetches full metrics for all repos concurrently.
    3. Runs `prepare_pipeline` (clean + memory-optimise) then `score_repos`
       (z-score normalise → weighted Hype/Substance scores → verdict).
    4. Saves the processed DataFrame to `_data/api_data/` as an Excel file.
    5. Returns results sorted by Substance Score descending.

    **Errors**

    - `404` — no repositories found for the given language.
    """
    raw_records = await fetch_trending_repos(language=language, limit=limit)

    if not raw_records:
        raise HTTPException(
            status_code=404,
            detail=f"No trending repositories found for language '{language}'.",
        )

    df = prepare_pipeline(raw_records)
    save_pipeline_data(df, f"trending_{language}")
    scored = score_repos(df)

    ranked = sorted(scored, key=lambda r: r.substance_score, reverse=True)

    return {
        "language": language,
        "count": len(ranked),
        "results": [
            {
                "rank": idx + 1,
                "repo": r.repo,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "scores": {
                    "hype_score": r.hype_score,
                    "substance_score": r.substance_score,
                },
                "meta": {
                    "language": r.language,
                    "description": r.description,
                    "html_url": r.html_url,
                },
            }
            for idx, r in enumerate(ranked)
        ],
    }
