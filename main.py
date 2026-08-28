from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import time
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
connection = psycopg2.connect(DATABASE_URL)
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

# ── Database setup (created ONCE, no duplicates) ─────
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

Base.metadata.create_all(bind=engine)

# ── In-memory cache of the most recent reading ───────
# (fast path for /latest so we don't hit the DB on every poll)
latest_reading = {}
last_update_time = 0

# ── Sensor reading model (ESP32 → backend) ───────────
class SensorReading(BaseModel):
    tank_level_pct: float
    flow_rate: float
    total_litres: float
    ph_value: float
    turbidity: float
    pump_status: bool

@app.post('/sensor-reading')
def receive_reading(reading: SensorReading):
    global latest_reading, last_update_time
    latest_reading = reading.dict()
    last_update_time = time.time()

    # Persist every reading to the cloud database
    db = SessionLocal()
    try:
        db_reading = Reading(**reading.dict())
        db.add(db_reading)
        db.commit()
    finally:
        db.close()

    print(f"Received: {latest_reading}")
    return {"status": "received"}

@app.get('/latest')
def get_latest():
    if not latest_reading:
        return {"status": "no data yet"}
    seconds_since_update = time.time() - last_update_time
    return {
        **latest_reading,
        "seconds_since_update": round(seconds_since_update, 1),
        "device_online": seconds_since_update < 15  # offline if no data for 15s
    }

@app.get('/history')
def get_history(limit: int = 100):
    """Returns the most recent N readings from the database,
    oldest first, for use in Analytics charts."""
    db = SessionLocal()
    try:
        rows = db.query(Reading).order_by(Reading.id.desc()).limit(limit).all()
    finally:
        db.close()
    rows.reverse()  # oldest -> newest, easier for charting
    return [{
        "timestamp": r.timestamp.isoformat(),
        "tank_level_pct": r.tank_level_pct,
        "flow_rate": r.flow_rate,
        "total_litres": r.total_litres,
        "ph_value": r.ph_value,
        "turbidity": r.turbidity,
        "pump_status": r.pump_status,
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

# ── Endpoints ──────────────────────────────────────────
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
