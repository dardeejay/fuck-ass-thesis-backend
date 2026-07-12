"""
Owns weather-snapshot / suspension-label / model-version persistence.

Shares the same Postgres instance as the Next.js frontend's Prisma-managed
DB (see project-handa/docker-compose.yml), but not the same tables — this
module manages its own tables via SQLAlchemy directly (no migration
framework; `create_all` is sufficient for this small, additive schema) so
there's no collision with Prisma's `User` table or its migration history.
"""

import os
from datetime import datetime, date as date_type
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/project_handa"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class WeatherSnapshot(Base):
    __tablename__ = "weather_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date_type] = mapped_column(Date, nullable=False)
    rainfall: Mapped[float] = mapped_column(Float, nullable=False)
    tmax: Mapped[float] = mapped_column(Float, nullable=False)
    tmin: Mapped[float] = mapped_column(Float, nullable=False)
    rh: Mapped[float] = mapped_column(Float, nullable=False)
    wind_speed: Mapped[float] = mapped_column(Float, nullable=False)
    pagasa_level: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tcws_signal: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Computed server-side from tmax/rh at write time (see app/heat_index.py)
    # -- not yet a trainable feature (zero heat-driven suspension examples
    # exist yet), captured now so it's available once/if that changes.
    heat_index: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    heat_category: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SuspensionLabel(Base):
    __tablename__ = "suspension_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date_type] = mapped_column(Date, nullable=False, unique=True)
    pre_school: Mapped[bool] = mapped_column(Boolean, default=False)
    elementary: Mapped[bool] = mapped_column(Boolean, default=False)
    high_school: Mapped[bool] = mapped_column(Boolean, default=False)
    college: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    weather_snapshot_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("weather_snapshots.id"), nullable=True
    )
    weather_snapshot: Mapped[Optional[WeatherSnapshot]] = relationship()
    # Whether this label is part of the currently-LIVE model's training set.
    # Stays False until an admin promotes the staging model that included it.
    included_in_live_model: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def suspended(self) -> bool:
        return self.pre_school or self.elementary or self.high_school or self.college


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)  # "staging" | "live" | "archived"
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1: Mapped[float] = mapped_column(Float, nullable=False)
    n_days: Mapped[int] = mapped_column(Integer, nullable=False)
    n_suspensions: Mapped[int] = mapped_column(Integer, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    promoted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    promoted_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)


def init_db():
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()
