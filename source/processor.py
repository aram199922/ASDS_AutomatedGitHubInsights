"""
processor.py — Pandas-based data cleaning and memory optimization.

Transforms raw JSON dictionaries from ingestion.py into a clean, typed
DataFrame ready for statistical analysis. Every column is explicitly
downcast to the smallest safe numeric type to minimize RAM usage.

Memory optimization note: GitHub API responses contain integer counts
(stars, forks, issues) that fit comfortably in int32, and ratios/scores
that fit in float32. Storing them as the default int64/float64 wastes
exactly 2× the memory for no benefit at our data sizes.
"""

import pathlib
from datetime import datetime

import numpy as np
import pandas as pd

# Absolute path to _data/api_data/ resolved from this file's location so it
# works regardless of the working directory uvicorn is started from.
_API_DATA_DIR = pathlib.Path(__file__).parent.parent / "_data" / "api_data"


# Columns that represent raw integer counts from the API.
_INT_COLUMNS = ["stars", "forks", "open_issues", "total_issues",
                "contributor_count", "repo_age_days"]

# Derived float columns computed during cleaning.
_FLOAT_COLUMNS = ["star_velocity", "commit_consistency",
                  "issue_resolution_rate", "fork_substance_ratio"]


def build_dataframe(raw_records: list[dict]) -> pd.DataFrame:
    """
    Convert a list of raw repo metric dicts into a structured DataFrame.

    Args:
        raw_records: List of dictionaries as returned by ingestion.fetch_repo_data
                     or ingestion.fetch_trending_repos.

    Returns:
        A DataFrame with one row per repository and standardized columns.

    Raises:
        ValueError: If raw_records is empty.
    """
    if not raw_records:
        raise ValueError("Cannot build a DataFrame from an empty record list.")

    df = pd.DataFrame(raw_records)

    # Ensure every expected column exists even if the API returned nothing.
    for col in _INT_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    return df


def clean_repo_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw metrics and derive analytical features.

    Steps performed:
    1. Fill missing values with safe defaults.
    2. Derive star_velocity, commit_consistency, issue_resolution_rate,
       and fork_substance_ratio from raw counts.
    3. Drop the raw `weekly_commits` list column (already summarized).

    Args:
        df: Raw DataFrame from build_dataframe().

    Returns:
        A cleaned DataFrame with derived feature columns added.
    """
    df = df.copy()

    # Replace NaN in integer columns with 0.
    for col in _INT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # --- Derive analytical features ---

    age_weeks = (df["repo_age_days"] / 7).clip(lower=1)

    # Stars gained per week of the repo's life.
    df["star_velocity"] = df["stars"] / age_weeks

    # Commit consistency: mean weekly commits over the last 52 weeks.
    # High stddev = bursty activity (hype spike), low stddev = steady work.
    if "weekly_commits" in df.columns:
        df["commit_mean"] = df["weekly_commits"].apply(
            lambda x: float(np.mean(x)) if isinstance(x, list) and x else 0.0
        )
        df["commit_stddev"] = df["weekly_commits"].apply(
            lambda x: float(np.std(x)) if isinstance(x, list) and len(x) > 1 else 0.0
        )
        # Normalize stddev against mean to get a coefficient of variation.
        # A low value means commits are spread consistently; a high value means spikes.
        df["commit_consistency"] = df["commit_mean"] / (df["commit_stddev"] + 1.0)
        df = df.drop(columns=["weekly_commits", "commit_mean", "commit_stddev"])
    else:
        df["commit_consistency"] = 0.0

    # Fraction of issues that have been resolved — active maintainer signal.
    df["issue_resolution_rate"] = np.where(
        df["total_issues"] > 0,
        (df["total_issues"] - df["open_issues"]) / df["total_issues"],
        0.0,
    )

    # Forks per star — a high ratio means people are actually using/building on this.
    df["fork_substance_ratio"] = np.where(
        df["stars"] > 0,
        df["forks"] / df["stars"],
        0.0,
    )

    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast all numeric columns to the smallest safe data type.

    Converts int64 → int32 for count columns and float64 → float32 for
    ratio/score columns. On a 20-repo DataFrame this saves ~50 % of the
    numeric column memory; on large batch jobs the savings compound.

    Args:
        df: Cleaned DataFrame from clean_repo_metrics().

    Returns:
        The same DataFrame with downcast numeric dtypes.
    """
    df = df.copy()

    int_targets = [c for c in _INT_COLUMNS if c in df.columns]
    float_targets = [c for c in _FLOAT_COLUMNS if c in df.columns]

    # Also catch any other numeric columns created during cleaning.
    for col in df.select_dtypes(include=["int64"]).columns:
        if col not in int_targets:
            int_targets.append(col)
    for col in df.select_dtypes(include=["float64"]).columns:
        if col not in float_targets:
            float_targets.append(col)

    df[int_targets] = df[int_targets].astype(np.int32)
    df[float_targets] = df[float_targets].astype(np.float32)

    return df


def prepare_pipeline(raw_records: list[dict]) -> pd.DataFrame:
    """
    Full processing pipeline: build → clean → optimize memory.

    Convenience wrapper used by the API layer to run the entire
    processor in one call.

    Args:
        raw_records: Raw list of repo metric dicts from ingestion.

    Returns:
        A memory-optimized, feature-rich DataFrame ready for analysis.
    """
    df = build_dataframe(raw_records)
    df = clean_repo_metrics(df)
    df = optimize_memory(df)
    return df


def save_pipeline_data(df: pd.DataFrame, label: str) -> None:
    """
    Persist a pipeline DataFrame to ``_data/api_data/`` as an Excel workbook.

    Filename format: ``{label}_{YYYYMMDD_HHMMSS}.xlsx``

    The ``label`` is used as a human-readable prefix so files are easy to
    identify at a glance (e.g. ``analyze_owner_repo_20260511_183000.xlsx`` or
    ``trending_python_20260511_183000.xlsx``).

    The directory is created automatically if it does not exist.  Any I/O
    error is swallowed silently so a disk issue never crashes the API.

    Args:
        df: The processed DataFrame to save (output of prepare_pipeline).
        label: Short descriptive prefix for the filename.
    """
    try:
        _API_DATA_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitise label so it is safe as a filename component.
        safe_label = label.replace("/", "_").replace("\\", "_").replace(" ", "_")
        path = _API_DATA_DIR / f"{safe_label}_{timestamp}.xlsx"
        df.to_excel(path, index=False)
    except Exception:
        # Never let a save failure propagate to the HTTP response.
        pass