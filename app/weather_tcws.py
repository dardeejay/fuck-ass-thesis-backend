"""
Scrapes PAGASA's Tropical Cyclone Wind Signal (TCWS) bulletin — a second,
separate warning system from the rainfall warnings in weather_pagasa.py.

This matters because TCWS triggers AUTOMATIC, non-discretionary class
cancellation, independent of rainfall. Per DepEd Order No. 37, s. 2022
(Enclosure, Sec. IV.1 "Typhoon"), in-person and online classes and work from
Kindergarten to Grade 12 and ALS are automatically cancelled in LGUs under
TCWS 1, 2, 3, 4, or 5 — the same rule for every signal level. College
classes are outside DepEd's scope (SUCs/LUCs may opt to follow the Order),
so the note points to CHED/school announcements for those.
Source: https://www.deped.gov.ph/wp-content/uploads/2022/09/DO_s2022_037.pdf

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


# Policy note, not an ML-derived probability. DepEd Order No. 37, s. 2022
# applies one rule to every TCWS level (see module docstring).
AUTOMATIC_SUSPENSION_NOTE = (
    "Signal No. {signal} in effect: in-person and online classes from Kindergarten to Grade 12 "
    "and ALS are automatically cancelled (DepEd Order No. 37, s. 2022). College classes follow "
    "CHED/school announcements."
)


def automatic_suspension_note(signal: Optional[int]) -> Optional[str]:
    if signal is None:
        return None
    if 1 <= signal <= 5:
        return AUTOMATIC_SUSPENSION_NOTE.format(signal=signal)
    return f"Signal No. {signal} in effect."
