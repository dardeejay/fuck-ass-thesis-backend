"""
Heat index — unlike weather_tcws.py, this needs NO new scraping. Heat index
is a direct function of temperature + relative humidity, both of which are
already model features (TMAX, RH) pulled from OpenWeatherMap/climatology in
app/main.py. PAGASA's own Heat Index product (pagasa.dost.gov.ph/weather/
heat-index) turned out to be a map/image widget (via panahon.gov.ph iframes),
not scrapable station-level text like the rainfall/TCWS bulletins were — so
this computes the value locally using the standard NWS Rothfusz regression
(the same formula PAGASA's own methodology is based on) and classifies it
using PAGASA's own published category bands (scraped from the
panahon.gov.ph widget backing pagasa.dost.gov.ph/weather/heat-index):
    Not Hazardous: <27C | Caution: 27-32C | Extreme Caution: 33-41C
    Danger: 42-51C | Extreme Danger: >=52C

IMPORTANT CAVEAT: those bands are PAGASA's general public health-risk
categories, not an education-specific policy. Unlike TCWS (where DepEd/
NDRRMC's automatic-suspension tiers per signal number are well-established),
there is no similarly authoritative, easily-citable rule tying heat index
bands directly to automatic class suspension at the time this was written —
verify current DepEd heat advisory guidance before citing exact thresholds.
The note below is a reasonable placeholder, not a verified policy citation.

Like TCWS, this is NOT added to suspension_model.py's FEATURE_COLS: the
historical training data has zero heat-driven suspension examples (every
scraped record is rainfall/advisory-driven), so there's nothing for the
model to learn a relationship from yet. It's captured on every weather
snapshot going forward (see app/db.py), so if/when a heat-driven suspension
gets logged through the labeling UI, it becomes available as a real feature
option — see suspension_model.py FEATURE_COLS for where to add it later.
"""

from typing import Optional

# (category, upper bound in Celsius, exclusive) -- last entry has no upper bound
_BANDS = [
    ("Not Hazardous", 27.0),
    ("Caution", 33.0),
    ("Extreme Caution", 42.0),
    ("Danger", 52.0),
]
_EXTREME_DANGER = "Extreme Danger"

# Placeholder guidance -- verify against current DepEd heat advisory policy
# before citing in the paper (see module docstring caveat).
_SUSPENSION_NOTE = {
    "Danger": "Heat index in the Danger range: consider limiting outdoor activities / alternative learning modes per current DepEd heat advisory guidance.",
    "Extreme Danger": "Heat index in the Extreme Danger range: high risk of heat-related illness; verify current DepEd guidance on suspension of face-to-face classes.",
}


def celsius_to_fahrenheit(c: float) -> float:
    return c * 9 / 5 + 32


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32) * 5 / 9


def _heat_index_fahrenheit(temp_f: float, rh: float) -> float:
    """NWS Rothfusz regression, with the standard low-temperature and
    high-humidity adjustment terms."""
    if temp_f < 80:
        simple = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (rh * 0.094))
        return (simple + temp_f) / 2 if simple >= 80 else simple

    hi = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * rh
        - 0.22475541 * temp_f * rh
        - 0.00683783 * temp_f ** 2
        - 0.05481717 * rh ** 2
        + 0.00122874 * temp_f ** 2 * rh
        + 0.00085282 * temp_f * rh ** 2
        - 0.00000199 * temp_f ** 2 * rh ** 2
    )

    if rh < 13 and 80 <= temp_f <= 112:
        hi -= ((13 - rh) / 4) * (((17 - abs(temp_f - 95)) / 17) ** 0.5)
    elif rh > 85 and 80 <= temp_f <= 87:
        hi += ((rh - 85) / 10) * ((87 - temp_f) / 5)

    return hi


def compute_heat_index(temp_celsius: float, rh: float) -> float:
    """Heat index in Celsius, given air temperature (Celsius) and relative humidity (%)."""
    hi_f = _heat_index_fahrenheit(celsius_to_fahrenheit(temp_celsius), rh)
    return fahrenheit_to_celsius(hi_f)


def classify_heat_index(heat_index_celsius: float) -> str:
    for category, upper_bound in _BANDS:
        if heat_index_celsius < upper_bound:
            return category
    return _EXTREME_DANGER


def suspension_note(category: str) -> Optional[str]:
    return _SUSPENSION_NOTE.get(category)
