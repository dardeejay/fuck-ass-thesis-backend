"""
Merges the static CSV-based historical dataset with admin-submitted live
labels (see app/db.py) for model training.

Two views are exposed:
  - live_training_data(): only labels already promoted into the live model
    (included_in_live_model=True) — this is what actually serves predictions.
  - staging_training_data(): ALL labels, promoted or not — used to preview
    what a retrain would look like before an admin promotes it.

Caveat: DB-sourced rows are sporadic single-day submissions with no
preceding-day weather history captured, so RAIN_3DAY_SUM/RAIN_7DAY_SUM can't
be computed as true rolling sums for them the way the CSV data allows.
They're approximated with same-day rainfall (the same fallback pattern
suggest_suspension() already uses) — acceptable since these rows are
temporally disconnected from the CSV data anyway (a ~1+ year gap), but worth
noting as a limitation if this ever gets written up.
"""

import pandas as pd
from sqlalchemy import select

import suspension_model as sm
from app.db import SuspensionLabel, get_session

_DB_FEATURE_COLS = ["DATE", "RAINFALL", "TMAX", "TMIN", "RH", "WIND_SPEED", "SUSPENDED"]
_ALL_COLS = ["DATE"] + sm.FEATURE_COLS + ["SUSPENDED"]


def _db_labels_to_frame(session, only_included: bool) -> pd.DataFrame:
    query = select(SuspensionLabel)
    if only_included:
        query = query.where(SuspensionLabel.included_in_live_model.is_(True))
    rows = session.execute(query).scalars().all()

    records = []
    for row in rows:
        snap = row.weather_snapshot
        if snap is None:
            continue  # can't build model features without a weather snapshot
        records.append({
            "DATE": pd.Timestamp(row.date),
            "RAINFALL": snap.rainfall,
            "TMAX": snap.tmax,
            "TMIN": snap.tmin,
            "RH": snap.rh,
            "WIND_SPEED": snap.wind_speed,
            "SUSPENDED": int(row.suspended),
        })
    df = pd.DataFrame.from_records(records, columns=_DB_FEATURE_COLS)
    df["RAIN_3DAY_SUM"] = df["RAINFALL"]
    df["RAIN_7DAY_SUM"] = df["RAINFALL"]
    df["MONTH"] = df["DATE"].dt.month if not df.empty else pd.Series(dtype=int)
    return df


def _build_dataset(only_included: bool) -> pd.DataFrame:
    weather = sm.load_weather()
    suspensions = sm.load_suspensions()
    csv_dataset, _, _, _ = sm.build_labeled_dataset(
        weather, suspensions, start="2023-01-01", end="2024-12-31"
    )
    csv_features = sm.build_features(csv_dataset)[_ALL_COLS]

    session = get_session()
    try:
        db_frame = _db_labels_to_frame(session, only_included=only_included)
    finally:
        session.close()

    if db_frame.empty:
        return csv_features

    combined = pd.concat([csv_features, db_frame[_ALL_COLS]], ignore_index=True)
    return combined.drop_duplicates(subset="DATE", keep="last").sort_values("DATE").reset_index(drop=True)


def live_training_data() -> pd.DataFrame:
    return _build_dataset(only_included=True)


def staging_training_data() -> pd.DataFrame:
    return _build_dataset(only_included=False)


def pending_label_count() -> int:
    session = get_session()
    try:
        rows = session.execute(
            select(SuspensionLabel).where(SuspensionLabel.included_in_live_model.is_(False))
        ).scalars().all()
        return len(rows)
    finally:
        session.close()
