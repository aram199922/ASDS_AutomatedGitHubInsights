"""
analyzer.py — Statistical scoring engine using scipy.

Converts clean DataFrames into human-readable verdicts by:
1. Z-score normalizing every metric so they share a common scale.
2. Applying domain-informed weights to separate "hype signals" from
   "substance signals."
3. Mapping the resulting scores to a final verdict: "Useful" or "Just Hype".

Why scipy.stats.zscore?
  Our metrics have incompatible units (stars in the thousands, ratios
  between 0 and 1). Z-score normalization rescales each column to
  mean=0, std=1, so no single metric dominates the composite score
  just because of its raw magnitude.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Weights for the two composite scores.
# Values sum to 1.0 within each group.
# ---------------------------------------------------------------------------

# Hype signals: things that inflate quickly without reflecting real usage.
_HYPE_WEIGHTS: dict[str, float] = {
    "star_velocity": 0.60,   # Fast-rising stars are the clearest hype signal.
    "stars": 0.40,           # Raw star count adds hype context.
}

# Substance signals: things that require ongoing human effort.
_SUBSTANCE_WEIGHTS: dict[str, float] = {
    "commit_consistency": 0.40,   # Steady commits = maintained project.
    "issue_resolution_rate": 0.35, # Resolved issues = responsive maintainers.
    "fork_substance_ratio": 0.15,  # Forks = people actively building on this.
    "contributor_count": 0.10,    # More contributors = community health.
}

# Thresholds for the final verdict (on a 0–10 scale).
_SUBSTANCE_USEFUL_THRESHOLD = 5.0
_HYPE_DOMINANCE_THRESHOLD = 6.5  # Hype score above this triggers extra scrutiny.


@dataclass
class RepoScore:
    """Holds all scoring outputs for a single repository."""
    repo: str
    hype_score: float
    substance_score: float
    verdict: str
    confidence: str
    language: str
    description: str
    html_url: str


def _zscore_dataframe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Apply z-score normalization to the specified columns in place.

    Single-row DataFrames (one repo analyzed alone) all get z=0 since
    there's no variance — we handle this edge case by keeping raw values
    clipped to a neutral range rather than crashing.

    Args:
        df: DataFrame containing the columns to normalize.
        columns: Column names to apply z-score to.

    Returns:
        A copy of the DataFrame with the specified columns replaced by
        their z-scores.
    """
    df = df.copy()
    available = [c for c in columns if c in df.columns]

    if len(df) == 1:
        # Can't compute z-scores across a single data point — default to 0.
        df[available] = 0.0
        return df

    z_scores = stats.zscore(df[available].astype(float), nan_policy="omit")
    df[available] = np.nan_to_num(z_scores, nan=0.0)
    return df


def _weighted_score(row: pd.Series, weights: dict[str, float]) -> float:
    """
    Compute a weighted sum of z-scored metrics and scale to 0–10.

    Args:
        row: A single DataFrame row (one repo's z-scored metrics).
        weights: Dict mapping column names to their weight (must sum to 1.0).

    Returns:
        A float score in the range [0, 10].
    """
    raw = sum(
        row.get(col, 0.0) * weight
        for col, weight in weights.items()
    )
    # Z-scores typically fall in [-3, +3]; map that to [0, 10].
    scaled = (raw + 3.0) / 6.0 * 10.0
    return float(np.clip(scaled, 0.0, 10.0))


def _assign_verdict(hype: float, substance: float) -> tuple[str, str]:
    """
    Convert numeric scores into a human-readable verdict and confidence label.

    Logic:
    - "Useful"    → substance is strong enough to stand on its own.
    - "Just Hype" → hype dominates and substance is weak.
    - Confidence is "High" when the gap between scores is large.

    Args:
        hype: Hype score in [0, 10].
        substance: Substance score in [0, 10].

    Returns:
        A tuple of (verdict, confidence) strings.
    """
    gap = abs(hype - substance)
    confidence = "High" if gap >= 2.5 else "Medium" if gap >= 1.0 else "Low"

    if substance > _SUBSTANCE_USEFUL_THRESHOLD:
        verdict = "Useful"
    elif hype >= _HYPE_DOMINANCE_THRESHOLD and substance <= _SUBSTANCE_USEFUL_THRESHOLD:
        verdict = "Just Hype"
    else:
        verdict = "Emerging"  # Too early to tell — neither score is decisive.

    return verdict, confidence


def score_repos(df: pd.DataFrame) -> list[RepoScore]:
    """
    Run the full scoring pipeline on a processed DataFrame.

    Steps:
    1. Z-score normalize all hype and substance metric columns.
    2. For each repo, compute weighted Hype Score and Substance Score.
    3. Assign a verdict and confidence level.

    Args:
        df: Memory-optimized DataFrame from processor.prepare_pipeline().

    Returns:
        A list of RepoScore dataclasses, one per repository row.
    """
    all_metric_cols = list(_HYPE_WEIGHTS.keys()) + list(_SUBSTANCE_WEIGHTS.keys())
    df_z = _zscore_dataframe(df, all_metric_cols)

    scores: list[RepoScore] = []
    for _, row in df_z.iterrows():
        hype = _weighted_score(row, _HYPE_WEIGHTS)
        substance = _weighted_score(row, _SUBSTANCE_WEIGHTS)
        verdict, confidence = _assign_verdict(hype, substance)

        scores.append(RepoScore(
            repo=str(row.get("repo", "unknown")),
            hype_score=round(hype, 2),
            substance_score=round(substance, 2),
            verdict=verdict,
            confidence=confidence,
            language=str(row.get("language", "Unknown")),
            description=str(row.get("description", "")),
            html_url=str(row.get("html_url", "")),
        ))

    return scores