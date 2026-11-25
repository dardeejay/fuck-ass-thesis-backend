"""
Scrapes PAGASA's Tropical Cyclone Wind Signal (TCWS) bulletin — a second,
separate warning system from the rainfall warnings in weather_pagasa.py.

This matters because TCWS triggers AUTOMATIC, non-discretionary class
suspension per DepEd/NDRRMC policy, independent of rainfall:
  - Signal No. 1: pre-school automatically suspended; other levels at
    LGU/school discretion.
  - Signal No. 2: pre-school and elementary automatically suspended.
  - Signal No. 3+: all levels automatically suspended.
(Verify the exact current DepEd Order wording before citing — these
thresholds have been revised before and may be revised again.)

The suspension_model.py training data contains ZERO signal-driven examples
(every scraped historical record was rainfall/advisory-driven), so this is
NOT modeled as an ML feature — there's nothing to learn a numeric
relationship from. Instead it's handled as a deterministic policy check,
applied on top of the ML output in app/main.py.
"""

import re
from dataclasses import dataclass
from typing import Optional

import httpx
from bs4 import BeautifulSoup

TCWS_URL = "https://pagasa.dost.gov.ph/tropical-cyclone/severe-weather-bulletin"


@dataclass
class TcwsStatus:
    signal: Optional[int]  # highest active signal number affecting the area, or None
    matched_areas_text: Optional[str]


def parse_tcws_html(html: str, location_keywords=("Olongapo", "Zambales")) -> TcwsStatus:
    # The signal number lives only in a CSS class (`signalno2`) / image filename —
    # NOT in the visible text — so split the raw HTML at each signal header
    # rather than relying on get_text() alone.
    markers = list(re.finditer(r'class="signalno(\d+)"', html))
    if not markers:
        return TcwsStatus(None, None)

    best_signal = None
    best_areas = None
    for i, m in enumerate(markers):
        signal_num = int(m.group(1))
        start = m.start()
        end = markers[i + 1].start() if i + 1 < len(markers) else start + 8000
        chunk_soup = BeautifulSoup(html[start:end], "html.parser")
        lines = [l.strip() for l in chunk_soup.get_text(separator="\n").split("\n") if l.strip()]

        try:
            luzon_idx = lines.index("Luzon")
            luzon_areas = lines[luzon_idx + 1]
        except (ValueError, IndexError):
            continue

        if luzon_areas == "-":
            continue
        if any(kw.lower() in luzon_areas.lower() for kw in location_keywords):
            if best_signal is None or signal_num > best_signal:
                best_signal = signal_num
                best_areas = luzon_areas

    return TcwsStatus(best_signal, best_areas)


def fetch_tcws_status(location_keywords=("Olongapo", "Zambales"), timeout: float = 4.0) -> TcwsStatus:
    resp = httpx.get(TCWS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    resp.raise_for_status()
    return parse_tcws_html(resp.text, location_keywords)


# DepEd/NDRRMC automatic-suspension tiers by signal number — verify current
# wording before citing in the paper; presented here as a policy note, not
# an ML-derived probability.
AUTOMATIC_SUSPENSION_NOTE = {
    1: "Signal No. 1 in effect: pre-school automatically suspended; other levels typically at LGU/school discretion.",
    2: "Signal No. 2 in effect: pre-school and elementary automatically suspended per DepEd policy.",
    3: "Signal No. 3 in effect: all levels automatically suspended per DepEd/NDRRMC policy.",
    4: "Signal No. 4 in effect: all levels automatically suspended; this is a high-end typhoon warning.",
    5: "Signal No. 5 in effect: all levels automatically suspended; catastrophic winds expected.",
}


def automatic_suspension_note(signal: Optional[int]) -> Optional[str]:
    if signal is None:
        return None
    return AUTOMATIC_SUSPENSION_NOTE.get(signal, f"Signal No. {signal} in effect.")
