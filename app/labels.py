"""CRUD for admin-submitted suspension labels and the weather snapshot
captured alongside each one."""

from datetime import date as date_type, datetime
from typing import Optional

from sqlalchemy import select

from app.db import SuspensionLabel, WeatherSnapshot, get_session
from app.heat_index import classify_heat_index, compute_heat_index


def upsert_label(
    label_date: date_type,
    pre_school: bool,
    elementary: bool,
    high_school: bool,
    college: bool,
    rainfall: float,
    tmax: float,
    tmin: float,
    rh: float,
    wind_speed: float,
    pagasa_level: Optional[str] = None,
    tcws_signal: Optional[int] = None,
    notes: Optional[str] = None,
    created_by: Optional[str] = None,
) -> dict:
    heat_index = compute_heat_index(tmax, rh)
    heat_category = classify_heat_index(heat_index)

    session = get_session()
    try:
        snapshot = WeatherSnapshot(
            date=label_date, rainfall=rainfall, tmax=tmax, tmin=tmin, rh=rh,
            wind_speed=wind_speed, pagasa_level=pagasa_level, tcws_signal=tcws_signal,
            heat_index=heat_index, heat_category=heat_category,
        )
        session.add(snapshot)
        session.flush()  # assigns snapshot.id

        existing = session.execute(
            select(SuspensionLabel).where(SuspensionLabel.date == label_date)
        ).scalar_one_or_none()

        if existing is not None:
            existing.pre_school = pre_school
            existing.elementary = elementary
            existing.high_school = high_school
            existing.college = college
            existing.notes = notes
            existing.weather_snapshot_id = snapshot.id
            existing.created_by = created_by
            existing.updated_at = datetime.utcnow()
            # Correcting an already-promoted label's values in place is a
            # data fix, not "new" data -- leave included_in_live_model as-is.
            label = existing
        else:
            label = SuspensionLabel(
                date=label_date, pre_school=pre_school, elementary=elementary,
                high_school=high_school, college=college, notes=notes,
                weather_snapshot_id=snapshot.id, created_by=created_by,
                included_in_live_model=False,
            )
            session.add(label)

        session.commit()
        session.refresh(label)
        return _label_to_dict(label)
    finally:
        session.close()


def list_labels() -> list:
    session = get_session()
    try:
        labels = session.execute(
            select(SuspensionLabel).order_by(SuspensionLabel.date.desc())
        ).scalars().all()
        return [_label_to_dict(l) for l in labels]
    finally:
        session.close()


def _label_to_dict(label: SuspensionLabel) -> dict:
    snap = label.weather_snapshot
    return {
        "id": label.id,
        "date": label.date,
        "pre_school": label.pre_school,
        "elementary": label.elementary,
        "high_school": label.high_school,
        "college": label.college,
        "suspended": label.suspended,
        "notes": label.notes,
        "included_in_live_model": label.included_in_live_model,
        "created_by": label.created_by,
        "created_at": label.created_at,
        "weather_snapshot": None if snap is None else {
            "rainfall": snap.rainfall,
            "tmax": snap.tmax,
            "tmin": snap.tmin,
            "rh": snap.rh,
            "wind_speed": snap.wind_speed,
            "pagasa_level": snap.pagasa_level,
            "tcws_signal": snap.tcws_signal,
            "heat_index": snap.heat_index,
            "heat_category": snap.heat_category,
        },
    }
