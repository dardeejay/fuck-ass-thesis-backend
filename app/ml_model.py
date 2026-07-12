"""
Trains and caches the suspension-recommendation model.

Uses the regularized Random Forest (max_depth=4, min_samples_leaf=4) —
leave-one-out CV showed it outperforms the default (unconstrained) RF,
Gradient Boosting, and Logistic Regression on this dataset (F1=0.55 vs.
0.44-0.49; see suspension_model.py's REGULARIZED_MODEL_FACTORIES). Training
takes well under a second on this dataset size, so it's retrained fresh
from data rather than persisted as a pickle — no benefit to caching a model
this cheap to reproduce, and it avoids version-skew between a saved pickle
and the current feature-engineering code.

Nightly retraining flow (see app/scheduler.py):
  - retrain_staging_and_record() evaluates what the model WOULD look like if
    it included every not-yet-promoted admin-submitted label, and stores
    those metrics as a "staging" ModelVersion row for an admin to review.
  - promote() is the only thing that actually changes what the live model
    is trained on — it flips pending labels to included_in_live_model=True
    and busts the model cache. Until promoted, new labels have zero effect
    on live predictions, by design (see the "manual promotion gate"
    decision — retraining nightly is cheap, but blindly swapping the live
    model on one new label is risky when the entire positive class is only
    ~24-30 examples).
"""

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Optional

import pandas as pd
from sqlalchemy import select

import suspension_model as sm
from app import training_data
from app.db import ModelVersion, SuspensionLabel, get_session

MODEL_NAME = "Random Forest (regularized)"


@dataclass
class Prediction:
    probability: float
    risk_level: str
    recommendation: str


@lru_cache(maxsize=1)
def get_model():
    features = training_data.live_training_data()
    X = features[sm.FEATURE_COLS]
    y = features["SUSPENDED"]

    model = sm.REGULARIZED_MODEL_FACTORIES[MODEL_NAME](random_state=42)
    model.fit(X, y)
    return model


def predict(
    rainfall: float,
    tmax: float,
    tmin: float,
    rh: float,
    wind_speed: float,
    month: int,
    rain_3day_sum: Optional[float] = None,
    rain_7day_sum: Optional[float] = None,
) -> Prediction:
    model = get_model()
    label, proba = sm.suggest_suspension(
        model, rainfall=rainfall, tmax=tmax, tmin=tmin, rh=rh,
        wind_speed=wind_speed, month=month,
        rain_3day_sum=rain_3day_sum, rain_7day_sum=rain_7day_sum,
    )
    risk_level = next(
        text for threshold, text in [(0.60, "high"), (0.30, "moderate"), (0.0, "low")]
        if proba >= threshold
    )
    return Prediction(probability=proba, risk_level=risk_level, recommendation=label)


def feature_importance() -> pd.Series:
    return sm.feature_importance(get_model())


def _evaluate(features: pd.DataFrame) -> dict:
    factory = {MODEL_NAME: sm.REGULARIZED_MODEL_FACTORIES[MODEL_NAME]}
    result = sm.leave_one_out_cv(features, model_factories=factory)
    row = result.iloc[0]
    return {
        "accuracy": float(row["accuracy"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "n_days": len(features),
        "n_suspensions": int(features["SUSPENDED"].sum()),
    }


@lru_cache(maxsize=1)
def metrics() -> dict:
    """LOO-CV for the currently-live model. Cached lazily (not at startup)
    since LOO over ~730+ rows takes on the order of a minute."""
    evaluated = _evaluate(training_data.live_training_data())
    return {
        "model_name": MODEL_NAME,
        "evaluation_method": "leave-one-out cross-validation",
        **evaluated,
    }


def retrain_staging_and_record() -> dict:
    """Evaluate the model as it WOULD look with all pending labels included,
    and record it as a staging ModelVersion for admin review. Does not affect
    the live model — see promote()."""
    evaluated = _evaluate(training_data.staging_training_data())

    session = get_session()
    try:
        version = ModelVersion(
            model_name=MODEL_NAME,
            status="staging",
            trained_at=datetime.utcnow(),
            **evaluated,
        )
        session.add(version)
        session.commit()
        session.refresh(version)
        return _version_to_dict(version)
    finally:
        session.close()


def promote(version_id: int, promoted_by: Optional[str] = None) -> dict:
    """Mark all pending labels as included in the live model, archive the
    previous live version, and promote the given staging version. Busts the
    model/metrics cache so the next prediction reflects the new data."""
    session = get_session()
    try:
        version = session.get(ModelVersion, version_id)
        if version is None:
            raise ValueError(f"No model version with id={version_id}")
        if version.status != "staging":
            raise ValueError(f"Version {version_id} is not staging (status={version.status})")

        pending = session.execute(
            select(SuspensionLabel).where(SuspensionLabel.included_in_live_model.is_(False))
        ).scalars().all()
        for label in pending:
            label.included_in_live_model = True

        previous_live = session.execute(
            select(ModelVersion).where(ModelVersion.status == "live")
        ).scalars().all()
        for old in previous_live:
            old.status = "archived"

        version.status = "live"
        version.promoted_at = datetime.utcnow()
        version.promoted_by = promoted_by

        session.commit()
        session.refresh(version)
        result = _version_to_dict(version)
    finally:
        session.close()

    get_model.cache_clear()
    metrics.cache_clear()
    return result


def list_versions() -> list:
    session = get_session()
    try:
        versions = session.execute(
            select(ModelVersion).order_by(ModelVersion.trained_at.desc())
        ).scalars().all()
        return [_version_to_dict(v) for v in versions]
    finally:
        session.close()


def _version_to_dict(version: ModelVersion) -> dict:
    return {
        "id": version.id,
        "model_name": version.model_name,
        "status": version.status,
        "accuracy": version.accuracy,
        "precision": version.precision,
        "recall": version.recall,
        "f1": version.f1,
        "n_days": version.n_days,
        "n_suspensions": version.n_suspensions,
        "trained_at": version.trained_at,
        "promoted_at": version.promoted_at,
        "promoted_by": version.promoted_by,
    }
