"""
Recent observed rainfall, for the model's accumulated-rain features.

The model was trained on daily data where RAINFALL is a 24h total and
RAIN_3DAY_SUM / RAIN_7DAY_SUM are rolling sums of those totals — and
RAIN_3DAY_SUM is its most important feature. Live and forecast inputs only
give "now" (PAGASA's warning level, OpenWeatherMap's 3-hour steps), so the
previous days' rain has to come from somewhere. OpenWeatherMap's free tier has
no history, so this pulls the past week of hourly precipitation from
Open-Meteo (free, no API key).

Rain is represented as (end_time, mm) events — the mm fell in the period
ending at end_time — so observed hours and forecast steps can be summed over
the same rolling windows.
"""

from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Tuple

import httpx

from app.weather_openweather import OLONGAPO_COORDS

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

RainEvent = Tuple[datetime, float]  # (UTC end of period, mm in that period)


def fetch_observed_hourly_rain(timeout: float = 4.0) -> List[RainEvent]:
    """Past 7 days of hourly precipitation up to now (UTC-aware timestamps).
    Open-Meteo labels each hour with its end time ("sum of the preceding hour")."""
    resp = httpx.get(
        OPEN_METEO_URL,
        params={
            "latitude": OLONGAPO_COORDS["lat"],
            "longitude": OLONGAPO_COORDS["lon"],
            "hourly": "precipitation",
            "past_days": 7,
            "forecast_days": 1,
            "timezone": "UTC",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    now = datetime.now(timezone.utc)
    events = []
    for ts, mm in zip(hourly["time"], hourly["precipitation"]):
        end = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        if end <= now and mm is not None:
            events.append((end, float(mm)))
    return events


def rain_between(events: Iterable[RainEvent], start: datetime, end: datetime) -> float:
    """Total rain over the window (start, end], to 0.1 mm (the precision of
    the source data — avoids float noise like 0.30000000000000004)."""
    return round(sum(mm for ts, mm in events if start < ts <= end), 1)


def trailing_sums(events: Iterable[RainEvent], end: datetime) -> Tuple[float, float, float]:
    """(24h, 72h, 168h) rain totals ending at `end` — the model's RAINFALL,
    RAIN_3DAY_SUM and RAIN_7DAY_SUM equivalents."""
    events = list(events)
    return tuple(rain_between(events, end - timedelta(hours=h), end) for h in (24, 72, 168))
