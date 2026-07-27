"""
Project Handa API — serves the class-suspension recommendation model to the
project-handa (Next.js) frontend.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

import datetime
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app import db, history, labels as labels_module, ml_model, scheduler  # noqa: E402
from app.schemas import (  # noqa: E402
    ForecastResponse,
    ForecastStepResponse,
    HeatIndexResponse,
    LabelCreateRequest,
    LabelResponse,
    LiveConditionsSource,
    LivePredictionResponse,
    ModelMetrics,
    ModelVersionResponse,
    PagasaWarningResponse,
    PredictionRequest,
    PredictionResponse,
    PromoteRequest,
    SuspensionRecord,
    TcwsStatusResponse,
)
from app.heat_index import classify_heat_index, compute_heat_index, suspension_note as heat_suspension_note  # noqa: E402
from app.weather_openweather import fetch_forecast, fetch_live_conditions  # noqa: E402
from app.weather_pagasa import fetch_pagasa_warning  # noqa: E402
from app.weather_tcws import automatic_suspension_note, fetch_tcws_status  # noqa: E402
from app.climatology import climatology_for_month  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(
    title="Project Handa API",
    description="Class suspension recommendation model for Olongapo City",
    version="0.1.0",
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _feature_importance_list():
    return [{"feature": k, "importance": float(v)} for k, v in ml_model.feature_importance().items()]


def _heat_index_response(tmax: float, rh: float) -> HeatIndexResponse:
    heat_index_c = compute_heat_index(tmax, rh)
    category = classify_heat_index(heat_index_c)
    return HeatIndexResponse(
        heat_index_celsius=heat_index_c, category=category, note=heat_suspension_note(category),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/predictions", response_model=PredictionResponse)
def create_prediction(body: PredictionRequest):
    result = ml_model.predict(
        rainfall=body.rainfall, tmax=body.tmax, tmin=body.tmin, rh=body.rh,
        wind_speed=body.wind_speed, month=body.month,
        rain_3day_sum=body.rain_3day_sum, rain_7day_sum=body.rain_7day_sum,
    )
    return PredictionResponse(
        probability=result.probability,
        risk_level=result.risk_level,
        recommendation=result.recommendation,
        model_name=ml_model.MODEL_NAME,
        feature_importance=_feature_importance_list(),
        heat_index=_heat_index_response(body.tmax, body.rh),
    )


@app.get("/v1/weather/pagasa", response_model=PagasaWarningResponse)
def get_pagasa_warning():
    try:
        warning = fetch_pagasa_warning()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach PAGASA: {exc}")
    return PagasaWarningResponse(
        level=warning.level,
        rainfall_mm_estimate=warning.rainfall_mm_estimate,
        matched_areas_text=warning.matched_areas_text,
        bulletin_title=warning.bulletin_title,
        issued_at=warning.issued_at,
    )


@app.get("/v1/weather/tcws", response_model=TcwsStatusResponse)
def get_tcws_status():
    try:
        status = fetch_tcws_status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach PAGASA: {exc}")
    return TcwsStatusResponse(
        signal=status.signal,
        matched_areas_text=status.matched_areas_text,
        automatic_suspension_note=automatic_suspension_note(status.signal),
    )


@app.get("/v1/predictions/live", response_model=LivePredictionResponse)
def create_live_prediction(month: Optional[int] = None):
    month = month or datetime.date.today().month

    try:
        warning = fetch_pagasa_warning()
        rainfall_source = "pagasa"
    except Exception:
        warning = None
        rainfall_source = "pagasa-fallback"

    rainfall = warning.rainfall_mm_estimate if warning else 5.0

    try:
        tcws = fetch_tcws_status()
    except Exception:
        tcws = None

    live = None
    try:
        live = fetch_live_conditions()
    except Exception:
        live = None

    if live is not None:
        tmax, tmin, rh, wind_speed = live.tmax, live.tmin, live.rh, live.wind_speed
        weather_source = "openweathermap"
    else:
        clim = climatology_for_month(month)
        tmax, tmin, rh, wind_speed = clim["tmax"], clim["tmin"], clim["rh"], clim["wind_speed"]
        weather_source = "climatology-fallback"

    result = ml_model.predict(
        rainfall=rainfall, tmax=tmax, tmin=tmin, rh=rh, wind_speed=wind_speed, month=month,
    )

    return LivePredictionResponse(
        probability=result.probability,
        risk_level=result.risk_level,
        recommendation=result.recommendation,
        model_name=ml_model.MODEL_NAME,
        feature_importance=_feature_importance_list(),
        heat_index=_heat_index_response(tmax, rh),
        pagasa_warning=PagasaWarningResponse(
            level=warning.level if warning else None,
            rainfall_mm_estimate=rainfall,
            matched_areas_text=warning.matched_areas_text if warning else None,
            bulletin_title=warning.bulletin_title if warning else None,
            issued_at=warning.issued_at if warning else None,
        ),
        tcws=TcwsStatusResponse(
            signal=tcws.signal if tcws else None,
            matched_areas_text=tcws.matched_areas_text if tcws else None,
            automatic_suspension_note=automatic_suspension_note(tcws.signal) if tcws else None,
        ),
        conditions_used=PredictionRequest(
            rainfall=rainfall, tmax=tmax, tmin=tmin, rh=rh, wind_speed=wind_speed, month=month,
        ),
        sources=LiveConditionsSource(rainfall_source=rainfall_source, weather_source=weather_source),
    )


@app.get("/v1/predictions/forecast", response_model=ForecastResponse)
def get_forecast_predictions():
    """Upcoming-hours strip: runs each 3-hour OpenWeatherMap forecast step
    through the same model as /v1/predictions/live. The model was validated
    on daily aggregates, not sub-daily steps — see ForecastResponse.source."""
    try:
        steps = fetch_forecast()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach OpenWeatherMap: {exc}")

    if steps is None:
        raise HTTPException(status_code=503, detail="OPENWEATHER_API_KEY not configured; forecast unavailable")

    results = []
    for step in steps:
        prediction = ml_model.predict(
            rainfall=step.rainfall_mm, tmax=step.tmax, tmin=step.tmin, rh=step.rh,
            wind_speed=step.wind_speed, month=step.timestamp.month,
        )
        results.append(ForecastStepResponse(
            timestamp=step.timestamp,
            tmax=step.tmax,
            tmin=step.tmin,
            rh=step.rh,
            wind_speed=step.wind_speed,
            rainfall=step.rainfall_mm,
            weather_id=step.weather_id,
            weather_main=step.weather_main,
            probability=prediction.probability,
            risk_level=prediction.risk_level,
            recommendation=prediction.recommendation,
        ))

    return ForecastResponse(
        model_name=ml_model.MODEL_NAME,
        generated_at=datetime.datetime.utcnow(),
        source=(
            "openweathermap-forecast, 3-hour steps; rainfall is OWM's own forecast "
            "(no PAGASA forecast exists) and the model was validated on daily "
            "conditions — treat these per-step suggestions as indicative, not a "
            "substitute for the daily recommendation above."
        ),
        steps=results,
    )


@app.get("/v1/history/suspensions", response_model=List[SuspensionRecord])
def get_suspension_history():
    return history.list_suspensions()


@app.get("/v1/model/metrics", response_model=ModelMetrics)
def get_model_metrics():
    return ml_model.metrics()


@app.post("/v1/labels", response_model=LabelResponse)
def create_label(body: LabelCreateRequest):
    return labels_module.upsert_label(
        label_date=body.date,
        pre_school=body.pre_school,
        elementary=body.elementary,
        high_school=body.high_school,
        college=body.college,
        rainfall=body.rainfall,
        tmax=body.tmax,
        tmin=body.tmin,
        rh=body.rh,
        wind_speed=body.wind_speed,
        pagasa_level=body.pagasa_level,
        tcws_signal=body.tcws_signal,
        notes=body.notes,
        created_by=body.created_by,
    )


@app.get("/v1/labels", response_model=List[LabelResponse])
def get_labels():
    return labels_module.list_labels()


@app.get("/v1/model/versions", response_model=List[ModelVersionResponse])
def get_model_versions():
    return ml_model.list_versions()


@app.post("/v1/model/retrain-staging", response_model=ModelVersionResponse)
def trigger_staging_retrain():
    """Manual trigger for the same job the nightly scheduler runs — useful for
    testing, or for retraining on demand right after submitting new labels
    instead of waiting for the 2 AM job."""
    return ml_model.retrain_staging_and_record()


@app.post("/v1/model/promote/{version_id}", response_model=ModelVersionResponse)
def promote_model_version(version_id: int, body: PromoteRequest):
    try:
        return ml_model.promote(version_id, promoted_by=body.promoted_by)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
