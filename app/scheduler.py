"""
Nightly staging-retrain job. This NEVER touches the live model -- it only
computes what the model would look like with pending labels included and
records it as a "staging" ModelVersion for an admin to review and promote
(see app/ml_model.py's module docstring for why promotion is a manual gate).
"""

import logging
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app import ml_model

logger = logging.getLogger("handa.scheduler")

_scheduler = BackgroundScheduler()

# The container runs in UTC by default (python:3.12-slim has no tzdata
# configured) -- without an explicit timezone here, "2 AM" would fire at
# 2 AM UTC = 10 AM Philippine time, defeating the point of running overnight.
MANILA_TZ = ZoneInfo("Asia/Manila")


def _nightly_retrain_job():
    try:
        result = ml_model.retrain_staging_and_record()
        logger.info("Nightly staging retrain recorded: %s", result)
    except Exception:
        logger.exception("Nightly staging retrain failed")


def start(hour: int = 2, minute: int = 0):
    _scheduler.add_job(
        _nightly_retrain_job,
        trigger=CronTrigger(hour=hour, minute=minute, timezone=MANILA_TZ),
        id="nightly-staging-retrain",
        replace_existing=True,
    )
    _scheduler.start()


def shutdown():
    _scheduler.shutdown(wait=False)
