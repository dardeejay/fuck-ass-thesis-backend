"""
Exploratory analysis + modeling scratch space for Project Handa.

Data:
  - "Cubi Point Daily Data.csv": daily weather (2022-2024) for
    Cubi Point (Subic Bay area) — RAINFALL(mm), TMAX, TMIN, RH, WIND_SPEED, WIND_DIRECTION.
  - "May-November 2025 Suspensions - Sheet1.csv": ground-truth class
    suspension decisions with the rainfall/forecast/time risk labels
    the mayor's office used when announcing them.

The two files don't overlap in time yet (weather data stops in 2024,
suspension labels start mid-2025), so there's nothing to train on
directly. Use this file to:
  1. Explore the weather data and flag candidate high-risk days for
     2022-2024 (to manually verify against the mayor's FB page and
     backfill suspension labels).
  2. Once backfilled labels exist for an overlapping period, plug them
     in at `load_suspensions()` and use `build_features()` /
     `train_baseline_model()` to fit a real model.

Run interactively (recommended): open in Jupyter/VS Code and run cell
by cell. Or `python analysis.py` for a quick console summary.
"""

import pandas as pd
import numpy as np

WEATHER_CSV = "Cubi Point Daily Data.csv"
SUSPENSIONS_CSV = "May-November 2025 Suspensions - Sheet1.csv"

# Rough PAGASA-style daily-rainfall thresholds (mm) used just to flag
# candidate suspension-worthy days for manual review — not a real model.
RAIN_THRESHOLDS = {
    "light": 7.5,
    "moderate": 15,
    "heavy": 30,
    "intense": 65,   # ballpark "orange warning" territory
    "torrential": 185,  # ballpark "red warning" territory
}


def load_weather(path=WEATHER_CSV):
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(dict(year=df.YEAR, month=df.MONTH, day=df.DAY))
    # -1 in this dataset marks trace rainfall (not missing) per PAGASA convention
    df["RAINFALL"] = df["RAINFALL"].clip(lower=0)
    return df.sort_values("DATE").reset_index(drop=True)


def load_suspensions(path=SUSPENSIONS_CSV):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df["Date Posted"] = pd.to_datetime(df["Date Posted"], format="%m/%d/%y")
    return df


def rain_category(mm):
    if mm >= RAIN_THRESHOLDS["torrential"]:
        return "torrential"
    if mm >= RAIN_THRESHOLDS["intense"]:
        return "intense"
    if mm >= RAIN_THRESHOLDS["heavy"]:
        return "heavy"
    if mm >= RAIN_THRESHOLDS["moderate"]:
        return "moderate"
    if mm >= RAIN_THRESHOLDS["light"]:
        return "light"
    return "none"


def flag_candidate_days(df, min_mm=30):
    """Days worth cross-checking against the mayor's FB page for a suspension announcement."""
    out = df.loc[df.RAINFALL >= min_mm, ["DATE", "RAINFALL", "TMAX", "TMIN", "RH", "WIND_SPEED"]].copy()
    out["CATEGORY"] = out.RAINFALL.map(rain_category)
    return out.sort_values("RAINFALL", ascending=False).reset_index(drop=True)


def months_to_review(df, min_mm=30):
    """Year-month buckets containing at least one candidate day, for quick FB-page lookup."""
    flagged = flag_candidate_days(df, min_mm=min_mm)
    flagged["YEAR_MONTH"] = flagged.DATE.dt.to_period("M")
    return (
        flagged.groupby("YEAR_MONTH")
        .agg(days=("DATE", "count"), max_rainfall_mm=("RAINFALL", "max"))
        .reset_index()
    )


def build_features(weather_df):
    """Feature frame keyed by DATE, ready to join against a labeled suspension column."""
    feats = weather_df.copy()
    feats["RAIN_CATEGORY"] = feats.RAINFALL.map(rain_category)
    feats["RAIN_3DAY_SUM"] = feats.RAINFALL.rolling(3, min_periods=1).sum()
    feats["RAIN_7DAY_SUM"] = feats.RAINFALL.rolling(7, min_periods=1).sum()
    return feats


def train_baseline_model(features_df, label_col="SUSPENDED"):
    """
    Fit once `features_df` has a binary label_col aligned to DATE
    (i.e. after backfilling 2022-2024 suspension history from the FB page,
    or once weather data for 2025 is added to match the labeled sheet).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    feature_cols = ["RAINFALL", "RAIN_3DAY_SUM", "RAIN_7DAY_SUM", "TMAX", "TMIN", "RH", "WIND_SPEED"]
    X = features_df[feature_cols]
    y = features_df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    return model


if __name__ == "__main__":
    weather = load_weather()
    print(weather.describe(include="all"))

    print("\nCandidate high-rainfall days to check against the mayor's FB page:")
    print(flag_candidate_days(weather, min_mm=65).to_string(index=False))

    print("\nMonths worth reviewing (>=30mm rain on at least one day):")
    print(months_to_review(weather, min_mm=30).to_string(index=False))

    print("\nExisting labeled suspensions (2025):")
    print(load_suspensions())
