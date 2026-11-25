"""
Replays real historical suspension days through the model — the simulator's
"what did the model say on a day classes were actually suspended?" check.

Only the 2023-2024 CSV days are used: they're the suspension days with a full
recorded weather row (incl. true 3-/7-day rain sums). Those days are also in
the live model's training data, so each replay is scored twice: by the live
model (which has seen the day) and by a held-out model refit without it,
matching the leave-one-out evaluation in the paper.
"""

from dataclasses import asdict
from datetime import date
from functools import lru_cache
from typing import List, Optional

import pandas as pd

from app import ml_model, training_data


@lru_cache(maxsize=1)
def _suspension_days() -> pd.DataFrame:
    features = training_data.csv_training_data()
    return features[features["SUSPENDED"] == 1].sort_values("DATE").reset_index(drop=True)


def _conditions(row) -> dict:
    return {
        "rainfall": float(row["RAINFALL"]),
        "tmax": float(row["TMAX"]),
        "tmin": float(row["TMIN"]),
        "rh": float(row["RH"]),
        "wind_speed": float(row["WIND_SPEED"]),
        "month": int(row["MONTH"]),
        "rain_3day_sum": round(float(row["RAIN_3DAY_SUM"]), 1),
        "rain_7day_sum": round(float(row["RAIN_7DAY_SUM"]), 1),
    }


def list_days() -> List[dict]:
    return [
        {"date": row["DATE"].date(), "conditions": _conditions(row)}
        for _, row in _suspension_days().iterrows()
    ]


def replay(day: date) -> Optional[dict]:
    """None if `day` isn't one of the recorded suspension days."""
    days = _suspension_days()
    match = days[days["DATE"] == pd.Timestamp(day)]
    if match.empty:
        return None

    conditions = _conditions(match.iloc[0])
    return {
        "date": day,
        "actually_suspended": True,
        "conditions": conditions,
        "live_model": asdict(ml_model.predict(**conditions)),
        "held_out": asdict(ml_model.predict(**conditions, model=ml_model.held_out_model(pd.Timestamp(day)))),
    }
