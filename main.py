from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import time
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# ── App must be created FIRST ────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load prediction model on startup ─────────────────
model = joblib.load('water_model.pkl')
pf    = joblib.load('water_poly_features.pkl')

# ── Database setup ────────────────────────────────────
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Reading(Base):
    __tablename__ = "readings"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    tank_level_pct = Column(Float)
    flow_rate = Column(Float)
    total_litres = Column(Float)
    ph_value = Column(Float)
    turbidity = Column(Float)
    pump_status = Column(Boolean)
    # Reconciliation metadata: was this reading uploaded live, or
    # recovered from the device's local buffer after an outage?
    source = Column(String, default="live")          # "live" | "reconciled"
    reconnect_event = Column(Integer, nullable=True)  # groups a batch together

# Logs each offline period, for evidence of offline-resilience testing
class ReconciliationEvent(Base):
    __tablename__ = "reconciliation_events"
    id = Column(Integer, primary_key=True, index=True)
    reconnect_event = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    downtime_ms = Column(Integer)
    readings_recovered = Column(Integer)

Base.metadata.create_all(bind=engine)

# ── In-memory cache of the most recent reading ───────
latest_reading = {}
last_update_time = 0

# ── Pending manual pump command (app -> device) ──────
# Advisory only: the device applies its own "physical state wins"
# rule and may reject this if stale or if a local safety fault is
# active. This is intentional - see firmware applyRemoteCommandIfValid().
pending_command = {"has_command": False, "pump_on": False, "issued_at": 0}

# ── Sensor reading model (ESP32 → backend) ───────────
class SensorReading(BaseModel):
    tank_level_pct: float
    flow_rate: float
    total_litres: float
    ph_value: float
    turbidity: float
    pump_status: bool

class BufferedReading(BaseModel):
    tank_level_pct: float
    flow_rate: float
    total_litres: float
    ph_value: float
    turbidity: float
    pump_status: bool

class SyncBatchRequest(BaseModel):
    reconnect_event: int
    downtime_ms: int
    readings: List[BufferedReading]

class PumpCommandRequest(BaseModel):
    pump_on: bool


@app.post('/sensor-reading')
def receive_reading(reading: SensorReading):
    global latest_reading, last_update_time
    latest_reading = reading.dict()
    last_update_time = time.time()

    db = SessionLocal()
    try:
        db_reading = Reading(**reading.dict(), source="live")
        db.add(db_reading)
        db.commit()
    finally:
        db.close()

    print(f"Received (live): {latest_reading}")
    return {"status": "received"}


@app.post('/sync-batch')
def sync_batch(batch: SyncBatchRequest):
    """
    Reconciliation endpoint. Called by the ESP32 immediately after it
    regains connectivity, carrying every reading it buffered locally
    while offline. This is the mechanism that resolves the split-brain
    problem: rather than the cloud record simply having a silent gap
    during an outage, the device's own local record - captured while
    running independently on edge logic - is uploaded and preserved,
    with the device's local timeline treated as authoritative.
    """
    global latest_reading, last_update_time

    db = SessionLocal()
    try:
        for r in batch.readings:
            db_reading = Reading(
                **r.dict(),
                source="reconciled",
                reconnect_event=batch.reconnect_event,
            )
            db.add(db_reading)

        event = ReconciliationEvent(
            reconnect_event=batch.reconnect_event,
            downtime_ms=batch.downtime_ms,
            readings_recovered=len(batch.readings),
        )
        db.add(event)
        db.commit()
    finally:
        db.close()

    # Bring the in-memory "latest" cache up to date with the most
    # recent reading recovered from the outage
    if batch.readings:
        latest_reading = batch.readings[-1].dict()
        last_update_time = time.time()

    print(f"[RECONCILE] Batch #{batch.reconnect_event}: "
          f"{len(batch.readings)} readings recovered after "
          f"{batch.downtime_ms/1000:.1f}s offline")

    return {
        "status": "reconciled",
        "readings_recovered": len(batch.readings),
        "downtime_seconds": round(batch.downtime_ms / 1000, 1),
    }


@app.get('/reconciliation-log')
def get_reconciliation_log(limit: int = 50):
    """Returns past offline/reconnect events - use this as evidence
    of offline-resilience for the evaluation chapter."""
    db = SessionLocal()
    try:
        rows = db.query(ReconciliationEvent).order_by(
            ReconciliationEvent.id.desc()).limit(limit).all()
    finally:
        db.close()
    return [{
        "reconnect_event": r.reconnect_event,
        "timestamp": r.timestamp.isoformat(),
        "downtime_seconds": round(r.downtime_ms / 1000, 1),
        "readings_recovered": r.readings_recovered,
    } for r in rows]


@app.post('/pump-command')
def issue_pump_command(cmd: PumpCommandRequest):
    """Called by the APP to request a manual pump override. This only
    queues the request - the device decides independently whether to
    honour it, based on staleness and local safety state."""
    global pending_command
    pending_command = {
        "has_command": True,
        "pump_on": cmd.pump_on,
        "issued_at": time.time(),
    }
    return {"status": "command queued", "note": "device may reject if stale or unsafe"}


@app.get('/pump-command')
def poll_pump_command():
    """Called by the ESP32 to check for a pending manual command."""
    global pending_command
    if not pending_command["has_command"]:
        return {"has_command": False}

    result = {
        "has_command": True,
        "pump_on": pending_command["pump_on"],
        "issued_at_ms": int(pending_command["issued_at"] * 1000),
    }
    pending_command = {"has_command": False, "pump_on": False, "issued_at": 0}  # consume once
    return result


@app.get('/latest')
def get_latest():
    if not latest_reading:
        return {"status": "no data yet"}
    seconds_since_update = time.time() - last_update_time
    return {
        **latest_reading,
        "seconds_since_update": round(seconds_since_update, 1),
        "device_online": seconds_since_update < 15
    }


@app.get('/history')
def get_history(limit: int = 100):
    db = SessionLocal()
    try:
        rows = db.query(Reading).order_by(Reading.id.desc()).limit(limit).all()
    finally:
        db.close()
    rows.reverse()
    return [{
        "timestamp": r.timestamp.isoformat(),
        "tank_level_pct": r.tank_level_pct,
        "flow_rate": r.flow_rate,
        "total_litres": r.total_litres,
        "ph_value": r.ph_value,
        "turbidity": r.turbidity,
        "pump_status": r.pump_status,
        "source": r.source,
    } for r in rows]

# ── Prediction models ─────────────────────────────────
class PredictRequest(BaseModel):
    day: int
    tank_level_pct: float
    tank_capacity_liters: float = 600.0

class PredictResponse(BaseModel):
    predicted_liters: float
    days_remaining: float
    alert: bool
    recommendation: str

@app.get('/')
def root():
    return {'status': 'Smart Water Tank API running'}

@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    X = np.array([[req.day]])
    X_poly = pf.transform(X)
    predicted = model.predict(X_poly)[0]
    predicted = max(predicted, 0)

    current_liters = req.tank_capacity_liters * req.tank_level_pct
    days_remaining = current_liters / predicted if predicted > 0 else 999

    alert = days_remaining < 2
    recommendation = (
        'Activate pump tonight — tank critical within 2 days'
        if alert else
        f'Tank sufficient for {days_remaining:.1f} more days'
    )

    return PredictResponse(
        predicted_liters=round(predicted, 1),
        days_remaining=round(days_remaining, 2),
        alert=alert,
        recommendation=recommendation
    )

@app.get('/tank-status')
def tank_status(level_pct: float = 0.75, capacity: float = 1000):
    current = capacity * level_pct
    return {
        'tank_level_pct': level_pct,
        'current_liters': current,
        'capacity_liters': capacity
    }