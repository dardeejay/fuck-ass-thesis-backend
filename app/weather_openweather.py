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
from typing import Optional

import httpx

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OLONGAPO_COORDS = {"lat": 14.8292, "lon": 120.2828}


@dataclass
class LiveConditions:
    tmax: float
    tmin: float
    rh: float
    wind_speed: float


def openweather_api_key() -> Optional[str]:
    return os.environ.get("OPENWEATHER_API_KEY")


def fetch_live_conditions(timeout: float = 10.0) -> Optional[LiveConditions]:
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
