from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime

app = FastAPI()

# เปิด CORS ให้ Dashboard เรียกได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลด Model ครั้งเดียวตอนเริ่ม
model = joblib.load("rain_model.pkl")

class SensorData(BaseModel):
    temperature:  float
    humidity:     float
    light_lux:    float
    uv_raw:       float
    rain_raw:     float
    rain_lag:     float = 4095
    hum_lag:      float = 70
    rain_change:  float = 0
    hum_change:   float = 0

@app.get("/")
def root():
    return {"status": "ITHALA20 AI API running ✅"}

@app.post("/predict")
def predict(data: SensorData):
    now   = datetime.now()
    hour  = now.hour
    month = now.month

    features = [[
        data.temperature,
        data.humidity,
        data.light_lux,
        data.uv_raw,
        data.rain_raw,
        data.rain_lag,
        data.hum_lag,
        data.rain_change,
        data.hum_change,
        hour,
        month
    ]]

    # ทำนายโอกาสฝน
    prob       = model.predict_proba(features)[0][1]
    rain_prob  = round(prob * 100, 1)

    if rain_prob >= 70:
        rain_label = "🌧️ มีโอกาสฝนตกสูง"
        advice     = "ควรเตรียมร่มและหลีกเลี่ยงกิจกรรมกลางแจ้ง"
    elif rain_prob >= 40:
        rain_label = "🌦️ อาจมีฝนบางส่วน"
        advice     = "ควรเตรียมร่มติดตัวไว้"
    else:
        rain_label = "☀️ ไม่มีฝน"
        advice     = "สภาพอากาศเหมาะสำหรับกิจกรรมกลางแจ้ง"

    # ประมาณ UV Index จาก uv_raw
    uv_voltage = data.uv_raw * (3.3 / 4095.0)
    uv_index   = round(uv_voltage / 0.1, 1)

    return {
        "rain_probability": rain_prob,
        "rain_label":       rain_label,
        "advice":           advice,
        "uv_index":         uv_index,
        "forecast_for":     "1 ชั่วโมงข้างหน้า",
        "timestamp":        now.isoformat()
    }