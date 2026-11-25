"""
Scrapes PAGASA's live rainfall warning bulletin — the actual ground truth
Olongapo's suspension decisions are based on. There is no public PAGASA API
for this (PANaHON's dashboard requires auth; the tenday API is token-gated
and covers a different product). The regional-forecast page is public,
server-rendered HTML naming affected municipalities directly under each
color heading, so it's scraped directly.

Bulletins on this page are extremely irregular free text (see the module
tests) — this parser is intentionally narrow: it only trusts the most
recent bulletin block and only extracts the RED/ORANGE/YELLOW WARNING LEVEL
lines. Any other structure PAGASA might use in future bulletins falls back
to level=None rather than guessing.
"""

import re
from dataclasses import dataclass
from typing import Optional

import httpx
from bs4 import BeautifulSoup

PAGASA_URL = "https://pagasa.dost.gov.ph/regional-forecast/ncrprsd"
WARNING_LEVELS = ["RED", "ORANGE", "YELLOW"]

# Approximate 24h rainfall accumulation (mm) each warning level corresponds
# to, per PAGASA's own rainfall warning system — used as a numeric proxy
# fed into the ML model's RAINFALL feature when only the categorical level
# is available (see analysis.py RAIN_THRESHOLDS for the same bands).
LEVEL_TO_RAINFALL_MM = {
    "RED": 220.0,      # >185mm/24h
    "ORANGE": 125.0,    # 65-185mm/24h, midpoint
    "YELLOW": 47.5,     # 30-65mm/24h, midpoint
    None: 5.0,          # no active warning for the area -> light/no rain assumed
}


@dataclass
class PagasaWarning:
    level: Optional[str]          # "RED" | "ORANGE" | "YELLOW" | None
    rainfall_mm_estimate: float
    matched_areas_text: Optional[str]
    bulletin_title: Optional[str]
    issued_at: Optional[str]


def _latest_bulletin_lines(rainfall_div_text: str) -> list[str]:
    lines = [l.strip() for l in rainfall_div_text.split("\n") if l.strip()]
    header_idxs = [
        i for i, l in enumerate(lines)
        if re.match(r"^(Heavy Rainfall|Rainfall|General Flood) Warning", l, re.IGNORECASE)
    ]
    end = header_idxs[1] if len(header_idxs) > 1 else len(lines)
    return lines[:end]


def parse_pagasa_html(html: str, location_keywords=("Olongapo", "Zambales")) -> PagasaWarning:
    soup = BeautifulSoup(html, "html.parser")
    div = soup.find("div", id="rainfalls")
    if div is None:
        return PagasaWarning(None, LEVEL_TO_RAINFALL_MM[None], None, None, None)

    lines = _latest_bulletin_lines(div.get_text(separator="\n"))
    bulletin_title = lines[0] if lines else None
    issued_at = next(
        (l.split(":", 1)[1].strip() for l in lines if l.lower().startswith("issued at")), None
    )

    found = {}
    for l in lines:
        m = re.match(r"^(RED|ORANGE|YELLOW)\s+WARNING LEVEL:\s*(.+)$", l, re.IGNORECASE)
        if m:
            color, areas = m.group(1).upper(), m.group(2)
            if any(kw.lower() in areas.lower() for kw in location_keywords):
                found[color] = areas

    for level in WARNING_LEVELS:
        if level in found:
            return PagasaWarning(level, LEVEL_TO_RAINFALL_MM[level], found[level], bulletin_title, issued_at)
    return PagasaWarning(None, LEVEL_TO_RAINFALL_MM[None], None, bulletin_title, issued_at)


def fetch_pagasa_warning(location_keywords=("Olongapo", "Zambales"), timeout: float = 4.0) -> PagasaWarning:
    resp = httpx.get(PAGASA_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    resp.raise_for_status()
    return parse_pagasa_html(resp.text, location_keywords)
