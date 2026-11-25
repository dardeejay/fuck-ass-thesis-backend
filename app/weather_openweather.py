"""
Optional live temp/humidity/wind source. RAINFALL — by far the dominant
model feature — comes from the PAGASA warning scrape (weather_pagasa.py),
NOT from here: testing showed OpenWeatherMap's current-conditions endpoint
can under-report actual rain intensity compared to PAGASA's real warnings.
This client only supplies the minor features (temp/humidity/wind), and only
when OPENWEATHER_API_KEY is configured — otherwise callers should fall back
to app.climatology.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import httpx

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OLONGAPO_COORDS = {"lat": 14.8292, "lon": 120.2828}


@dataclass
class LiveConditions:
    tmax: float
    tmin: float
    rh: float
    wind_speed: float


@dataclass
class ForecastStep:
    timestamp: datetime  # UTC-aware
    tmax: float
    tmin: float
    rh: float
    wind_speed: float
    rainfall_mm: float
    weather_id: int
    weather_main: str


def openweather_api_key() -> Optional[str]:
    return os.environ.get("OPENWEATHER_API_KEY")


def fetch_live_conditions(timeout: float = 4.0) -> Optional[LiveConditions]:
    api_key = openweather_api_key()
    if not api_key:
        return None

    resp = httpx.get(
        OPENWEATHER_URL,
        params={**OLONGAPO_COORDS, "appid": api_key, "units": "metric"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    main = data["main"]

    # OWM's /weather endpoint reports wind in m/s under units=metric — matches
    # this project's training data scale directly, no conversion needed. Note
    # temp_max/temp_min on this endpoint just repeat the instantaneous temp,
    # not a true daily high/low (see the earlier live-conditions test).
    return LiveConditions(
        tmax=main.get("temp_max", main["temp"]),
        tmin=main.get("temp_min", main["temp"]),
        rh=main["humidity"],
        wind_speed=data.get("wind", {}).get("speed", 0.0),
    )


def fetch_forecast(timeout: float = 4.0) -> Optional[List[ForecastStep]]:
    """5-day/3-hour-step forecast for the dashboard's upcoming-hours strip.
    Unlike fetch_live_conditions() above, rainfall here comes straight from
    OpenWeatherMap's own forecast (rain."3h"), not PAGASA — PAGASA only
    publishes a current bulletin, there's no forecast API for it. So this
    forecast's rainfall figures are less trustworthy than the "live" endpoint's
    (which prefers PAGASA), by design of what data actually exists."""
    api_key = openweather_api_key()
    if not api_key:
        return None

    resp = httpx.get(
        OPENWEATHER_FORECAST_URL,
        params={**OLONGAPO_COORDS, "appid": api_key, "units": "metric"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    steps = []
    for entry in data.get("list", []):
        main = entry["main"]
        weather = (entry.get("weather") or [{}])[0]
        steps.append(
            ForecastStep(
                timestamp=datetime.fromtimestamp(entry["dt"], tz=timezone.utc),
                tmax=main.get("temp_max", main["temp"]),
                tmin=main.get("temp_min", main["temp"]),
                rh=main["humidity"],
                wind_speed=entry.get("wind", {}).get("speed", 0.0),
                rainfall_mm=entry.get("rain", {}).get("3h", 0.0),
                weather_id=weather.get("id", 800),
                weather_main=weather.get("main", "Clear"),
            )
        )
    return steps
