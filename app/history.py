"""Serves the scraped historical suspension records for the frontend's
historical-records dashboard page (currently hardcoded mock data there)."""

from functools import lru_cache
from typing import List, Optional

import pandas as pd

import suspension_model as sm


def _to_bool_or_none(value) -> Optional[bool]:
    if pd.isna(value):
        return None
    return bool(value)


@lru_cache(maxsize=1)
def _suspensions() -> pd.DataFrame:
    return sm.load_suspensions()


def list_suspensions() -> List[dict]:
    df = _suspensions()
    records = []
    for _, row in df.sort_values("Date", ascending=False).iterrows():
        records.append({
            "date": row["Date"].date(),
            "rainfall_level": row["Rainfall Level"] if pd.notna(row["Rainfall Level"]) else None,
            "forecast_risk": row["Forecast Risk"] if pd.notna(row["Forecast Risk"]) else None,
            "time_risk": row["Time Risk"] if pd.notna(row["Time Risk"]) else None,
            "suspended": bool(row["Suspension Decision"]),
            "pre_school": _to_bool_or_none(row["Pre-School"]),
            "elementary": _to_bool_or_none(row["Elementary"]),
            "high_school": _to_bool_or_none(row["High School"]),
            "college": _to_bool_or_none(row["College"]),
        })
    return records
