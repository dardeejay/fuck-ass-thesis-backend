"""
Monthly climatological averages (TMAX/TMIN/RH/WIND_SPEED) computed from the
Cubi Point historical dataset. Used as a fallback for those features when no
live weather API key is configured — RAINFALL (by far the dominant feature,
see feature_importance() in suspension_model.py) instead comes from the live
PAGASA warning scrape, so this fallback only affects the minor features.
"""

from functools import lru_cache

import pandas as pd

from suspension_model import load_weather


@lru_cache(maxsize=1)
def monthly_climatology() -> pd.DataFrame:
    weather = load_weather()
    return weather.groupby(weather.DATE.dt.month)[["TMAX", "TMIN", "RH", "WIND_SPEED"]].mean()


def climatology_for_month(month: int) -> dict:
    row = monthly_climatology().loc[month]
    return {
        "tmax": float(row["TMAX"]),
        "tmin": float(row["TMIN"]),
        "rh": float(row["RH"]),
        "wind_speed": float(row["WIND_SPEED"]),
    }
