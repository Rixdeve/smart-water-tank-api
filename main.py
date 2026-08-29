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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('water_model.pkl')
pf    = joblib.load('water_poly_features.pkl')

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
    source = Column(String, default="live")
    reconnect_event = Column(Integer, nullable=True)

class ReconciliationEvent(Base):
    __tablename__ = "reconciliation_events"
    id = Column(Integer, primary_key=True, index=True)
    reconnect_event = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    downtime_ms = Column(Integer)
    readings_recovered = Column(Integer)

Base.metadata.create_all(bind=engine)

latest_reading = {}
last_update_time = 0

TANK_SPEC_TABLE = {
    550:   1175,
    750:   1180,
    1000:  1470,
    2000:  1688,
    3000:  1740,
    5000:  1810,
    10000: 2840,
}

SENSOR_MOUNT_MARGIN_CM = 5.0


def estimate_empty_distance_cm(capacity_liters: float) -> float:
    """Interpolates/extrapolates expected tank height from the spec
    table for a given capacity, and converts it to an estimated
    sensor-to-bottom (empty) distance in cm."""
    capacities = sorted(TANK_SPEC_TABLE.keys())

    if capacity_liters <= capacities[0]:
        height_mm = TANK_SPEC_TABLE[capacities[0]]
    elif capacity_liters >= capacities[-1]:
        height_mm = TANK_SPEC_TABLE[capacities[-1]]
    else:
        lower = max(c for c in capacities if c <= capacity_liters)
        upper = min(c for c in capacities if c >= capacity_liters)
        if lower == upper:
            height_mm = TANK_SPEC_TABLE[lower]
        else:
            frac = (capacity_liters - lower) / (upper - lower)
            height_mm = TANK_SPEC_TABLE[lower] + frac * (TANK_SPEC_TABLE[upper] - TANK_SPEC_TABLE[lower])

    height_cm = height_mm / 10.0
    return round(height_cm - SENSOR_MOUNT_MARGIN_CM, 1)


device_config = {
    "tank_capacity_liters": 650.0,
    "min_distance_cm": 22.0,
    "max_distance_cm": 400.0,
    "estimated_empty_distance_cm": estimate_empty_distance_cm(650.0),
    "empty_distance_manually_calibrated": False,
}

class DeviceConfigRequest(BaseModel):
    tank_capacity_liters: Optional[float] = None
    min_distance_cm: Optional[float] = None
    max_distance_cm: Optional[float] = None


@app.post('/device-config')
def set_device_config(cfg: DeviceConfigRequest):
    """Called by the APP to update tank capacity (and, if needed,
    advanced sensor limits). The ESP32 picks this up on its next
    config fetch - it is never pushed directly, keeping the cloud
    record as the single source of truth.

    Whenever capacity changes, the estimated empty-distance is
    recomputed from the manufacturer spec table. This estimate is
    only ever used by the device if it has not yet been manually
    calibrated - a real measurement always takes precedence."""
    if cfg.tank_capacity_liters is not None:
        device_config["tank_capacity_liters"] = cfg.tank_capacity_liters
        device_config["estimated_empty_distance_cm"] = estimate_empty_distance_cm(cfg.tank_capacity_liters)
    if cfg.min_distance_cm is not None:
        device_config["min_distance_cm"] = cfg.min_distance_cm
    if cfg.max_distance_cm is not None:
        device_config["max_distance_cm"] = cfg.max_distance_cm
    return {"status": "saved", "config": device_config}


@app.get('/device-config')
def get_device_config():
    """Called by the ESP32 on boot and periodically to sync capacity,
    sensor limits, and the manufacturer-spec-estimated empty distance.
    If unreachable, the device falls back to its own last-known/
    hardcoded defaults - this call is advisory, not a hard dependency,
    consistent with the offline-first design."""
    return device_config


@app.post('/device-config/mark-calibrated')
def mark_manually_calibrated():
    """Called by the ESP32 once the user performs a real
    /calibrate-empty reading, so the cloud stops advertising the
    manufacturer-spec estimate as authoritative for this device."""
    device_config["empty_distance_manually_calibrated"] = True
    return {"status": "marked"}


pending_command = {"has_command": False, "pump_on": False, "issued_at": 0}

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
    pending_command = {"has_command": False, "pump_on": False, "issued_at": 0}
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
