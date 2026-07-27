from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    rainfall: float = Field(..., ge=0, description="Daily rainfall in mm")
    tmax: float = Field(..., description="Daily max temperature, Celsius")
    tmin: float = Field(..., description="Daily min temperature, Celsius")
    rh: float = Field(..., ge=0, le=100, description="Relative humidity, %")
    wind_speed: float = Field(..., ge=0, description="Wind speed, m/s")
    month: int = Field(..., ge=1, le=12)
    rain_3day_sum: Optional[float] = Field(None, ge=0, description="Trailing 3-day rainfall total, mm. Defaults to `rainfall` if omitted.")
    rain_7day_sum: Optional[float] = Field(None, ge=0, description="Trailing 7-day rainfall total, mm. Defaults to `rainfall` if omitted.")


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class HeatIndexResponse(BaseModel):
    heat_index_celsius: float
    category: str  # "Not Hazardous" | "Caution" | "Extreme Caution" | "Danger" | "Extreme Danger"
    note: Optional[str]  # see app/heat_index.py caveat — not a verified DepEd policy citation


class PredictionResponse(BaseModel):
    probability: float
    risk_level: str  # "low" | "moderate" | "high"
    recommendation: str
    model_name: str
    feature_importance: List[FeatureImportance]
    heat_index: HeatIndexResponse  # computed from tmax/rh — see app/heat_index.py; not an ML feature (yet)


class PagasaWarningResponse(BaseModel):
    level: Optional[str]  # "RED" | "ORANGE" | "YELLOW" | None
    rainfall_mm_estimate: float
    matched_areas_text: Optional[str]
    bulletin_title: Optional[str]
    issued_at: Optional[str]


class LiveConditionsSource(BaseModel):
    rainfall_source: str  # "pagasa" | "pagasa-fallback"
    weather_source: str   # "openweathermap" | "climatology-fallback"


class TcwsStatusResponse(BaseModel):
    signal: Optional[int]  # highest active Tropical Cyclone Wind Signal number, or None
    matched_areas_text: Optional[str]
    automatic_suspension_note: Optional[str]


class LivePredictionResponse(PredictionResponse):
    pagasa_warning: PagasaWarningResponse
    tcws: TcwsStatusResponse  # separate from the ML score — see weather_tcws.py docstring
    conditions_used: PredictionRequest
    sources: LiveConditionsSource


class SuspensionRecord(BaseModel):
    date: date
    rainfall_level: Optional[str]
    forecast_risk: Optional[str]
    time_risk: Optional[str]
    suspended: bool
    pre_school: Optional[bool]
    elementary: Optional[bool]
    high_school: Optional[bool]
    college: Optional[bool]


class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    evaluation_method: str
    n_days: int
    n_suspensions: int


class LabelCreateRequest(BaseModel):
    date: date
    pre_school: bool
    elementary: bool
    high_school: bool
    college: bool
    rainfall: float = Field(..., ge=0)
    tmax: float
    tmin: float
    rh: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0)
    pagasa_level: Optional[str] = None
    tcws_signal: Optional[int] = None
    notes: Optional[str] = None
    created_by: Optional[str] = None


class WeatherSnapshotResponse(BaseModel):
    rainfall: float
    tmax: float
    tmin: float
    rh: float
    wind_speed: float
    pagasa_level: Optional[str]
    tcws_signal: Optional[int]
    heat_index: Optional[float]
    heat_category: Optional[str]


class LabelResponse(BaseModel):
    id: int
    date: date
    pre_school: bool
    elementary: bool
    high_school: bool
    college: bool
    suspended: bool
    notes: Optional[str]
    included_in_live_model: bool
    created_by: Optional[str]
    created_at: datetime
    weather_snapshot: Optional[WeatherSnapshotResponse]


class ModelVersionResponse(BaseModel):
    id: int
    model_name: str
    status: str  # "staging" | "live" | "archived"
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_days: int
    n_suspensions: int
    trained_at: datetime
    promoted_at: Optional[datetime]
    promoted_by: Optional[str]


class PromoteRequest(BaseModel):
    promoted_by: Optional[str] = None


class ForecastStepResponse(BaseModel):
    timestamp: datetime  # UTC
    tmax: float
    tmin: float
    rh: float
    wind_speed: float
    rainfall: float
    weather_id: int
    weather_main: str
    probability: float
    risk_level: str  # "low" | "moderate" | "high"
    recommendation: str


class ForecastResponse(BaseModel):
    model_name: str
    generated_at: datetime
    # Deliberately verbose: this is the one place callers are told the model
    # was validated at daily granularity, not the 3-hour steps used here.
    source: str
    steps: List[ForecastStepResponse]
