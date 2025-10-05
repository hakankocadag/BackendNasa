import os
import math
import numpy as np
import pandas as pd
import httpx
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ----------------------------
# Env
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-preview-05-20")
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_API_URL = f"{GEMINI_API_BASE_URL}/{GEMINI_MODEL_NAME}:generateContent"
IMAGEN_MODEL_NAME = "imagen-3.0-generate-002"
IMAGEN_API_URL = f"{GEMINI_API_BASE_URL}/{IMAGEN_MODEL_NAME}:predict"

# ----------------------------
# Config & Data Loading
# ----------------------------
CSV_PATH = os.getenv("CSV_PATH", "NASA_dataset.csv")

try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
except FileNotFoundError:
    print(f"Error: CSV file not found at path specified: {CSV_PATH}.")
    raise RuntimeError(
        "CSV file not found. Please ensure the path in your .env file or the default fallback is correct."
    )

if "DATE" in df.columns:
    df["DATE"] = pd.to_datetime(df["DATE"])
else:
    if all(col in df.columns for col in ["YEAR", "MO", "DY"]):
        df["DATE"] = pd.to_datetime(dict(
            year=df["YEAR"].astype(int),
            month=df["MO"].astype(int),
            day=df["DY"].astype(int)
        ))
    else:
        raise RuntimeError("CSV must contain DATE or (YEAR, MO, DY) columns.")

REQUIRED = ["LATITUDE", "LONGITUDE", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
for col in REQUIRED:
    if col not in df.columns:
        raise RuntimeError(f"CSV missing required column: {col}")

if "CITY" not in df.columns:
    df["CITY"] = "Unknown"

df["DOY"] = df["DATE"].dt.dayofyear
print(f"Data loaded successfully. Rows: {len(df)}. Cities: {df['CITY'].nunique()}")

# ----------------------------
# Request/Response Models (Weather)
# ----------------------------
class QueryRequest(BaseModel):
    lat: Optional[float] = Field(None, description="Latitude in decimal degrees")
    lon: Optional[float] = Field(None, description="Longitude in decimal degrees")
    city: Optional[str] = Field(None, description="City name as in CSV")
    date: str = Field(..., description="Target date (YYYY-MM-DD)")
    window_days: int = Field(7, ge=1, le=30, description="Climatology window half-width in days")
    rain_mm_threshold: float = Field(1.0, ge=0.0, description="Daily precipitation threshold (mm)")
    windy_ms_threshold: float = Field(5.0, ge=0.0, description="Windy day threshold (m/s)")
    hot_c_threshold: float = Field(30.0, description="Hot day threshold (°C)")
    cold_c_threshold: float = Field(0.0, description="Cold day threshold (°C)")

class QueryResponse(BaseModel):
    location: Dict[str, Any]
    climatology_window: Dict[str, str]
    sample_size: int
    metrics: Dict[str, float]
    probabilities: Dict[str, float]
    friendly_advice: str
    debug: Dict[str, Any]

class ImageRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt used to generate the image.")
    style_hint: Optional[str] = Field("cinematic, photorealistic",
                                      description="Style to apply to the image generation.")

class ImageResponse(BaseModel):
    prompt: str
    base64_image: str = Field(..., description="Base64 encoded PNG image data.")

# ----------------------------
# Utils
# ----------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def pick_location(subdf: pd.DataFrame, lat: Optional[float], lon: Optional[float], city: Optional[str]) -> pd.DataFrame:
    if city:
        cands = subdf[subdf["CITY"].astype(str).str.lower() == str(city).lower()]
        if not cands.empty:
            return cands
    if lat is not None and lon is not None:
        centroids = (
            subdf.groupby("CITY")[["LATITUDE", "LONGITUDE"]]
            .mean().reset_index()
        )
        if centroids.empty:
            return subdf
        dists = centroids.apply(lambda r: haversine_km(lat, lon, r["LATITUDE"], r["LONGITUDE"]), axis=1)
        nearest_city = centroids.iloc[int(np.argmin(dists))]["CITY"]
        return subdf[subdf["CITY"] == nearest_city]
    return subdf

def doy_window_mask(doy_series: pd.Series, target_doy: int, half_width: int) -> pd.Series:
    dist = (doy_series - target_doy).abs()
    wrap = 365 - dist
    circ_dist = np.minimum(dist, wrap)
    return circ_dist <= half_width

def safe_mean(x: pd.Series) -> float:
    return float(np.nanmean(x.values)) if len(x) else float("nan")

def pct_true(x: pd.Series) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    return float(100.0 * (x.sum() / n))

# ----------------------------
# Gemini Integration (Text)
# ----------------------------
async def call_gemini_api(payload: Dict[str, Any]) -> str:
    max_retries = 3
    base_delay = 1.0
    headers = {}
    url = GEMINI_API_URL
    if GEMINI_API_KEY:
        headers["x-goog-api-key"] = GEMINI_API_KEY
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(url, json=payload, headers=headers or None)
                response.raise_for_status()
                result = response.json()
                text = (
                    result.get('candidates', [{}])[0]
                          .get('content', {})
                          .get('parts', [{}])[0]
                          .get('text', '')
                )
                if text:
                    return text.strip()
                else:
                    raise Exception("Gemini response was empty.")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    return "AI Advice unavailable due to API error: 403 (Authentication/Permission issue)."
                if attempt == max_retries - 1:
                    return f"AI Advice unavailable due to API error: {e.response.status_code}"
            except Exception:
                if attempt == max_retries - 1:
                    return "AI Advice unavailable due to an unexpected error."
            await asyncio.sleep(base_delay * (2 ** attempt))
    return "AI Advice generation failed after multiple retries."

async def get_ai_advice(city: str, metrics: Dict[str, float], probabilities: Dict[str, float], target_date: str) -> str:
    system_prompt = (
        "You are a highly skilled, friendly, and conversational Climatology Consultant "
        "specializing in event planning. Your goal is to provide a single, concise paragraph "
        "of actionable advice based on the provided long-term historical probability data. "
        "Do not use markdown, lists, or headers. Focus on what preparations the user should make. "
        "Keep the tone encouraging yet realistic."
    )

    user_query = f"""
The event is planned for {target_date} in {city}. Here are the calculated climatology statistics for the time window:

PROBABILITIES (Chance of event occurring, based on historical data):
- Chance of significant rain (over 1mm): {probabilities['rain_probability_pct']:.1f}%
- Chance of strong winds (over 5m/s): {probabilities['windy_probability_pct']:.1f}%
- Chance of a hot day (max temp over 30°C): {probabilities['hot_probability_pct']:.1f}%
- Chance of a cold day (min temp under 0°C): {probabilities['cold_probability_pct']:.1f}%

AVERAGE CONDITIONS (Historical averages for the time window):
- Mean Maximum Temperature: {metrics['mean_tmax_c']:.1f}°C
- Mean Minimum Temperature: {metrics['mean_tmin_c']:.1f}°C
- Mean Wind Speed: {metrics['mean_wind_ms']:.1f} m/s
- Mean Relative Humidity: {metrics['mean_rh_pct']:.1f}%

Please provide a single paragraph of tailored advice for planning.
    """

    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]},
        "contents": [
            {"role": "user", "parts": [{"text": user_query}]}
        ],
    }
    return await call_gemini_api(payload)

# ----------------------------
# Chat Models
# ----------------------------
class Msg(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatReq(BaseModel):
    messages: List[Msg]
    session_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class ChatRes(BaseModel):
    reply: str
    usage: Optional[Dict[str, Any]] = None

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Will It Rain on My Parade? - Climatology API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "rows": len(df),
        "cities": int(df["CITY"].nunique()) if "CITY" in df.columns else None
    }

# ----------------------------
# Chat Endpoint
# ----------------------------
@app.post("/chat", response_model=ChatRes)
async def chat(req: ChatReq):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    system_parts = []
    content_parts = []
    for m in req.messages:
        if m.role == "system":
            system_parts.append({"text": m.content})
        else:
            content_parts.append({
                "role": m.role,
                "parts": [{"text": m.content}]
            })

    payload: Dict[str, Any] = {"contents": content_parts}
    if system_parts:
        payload["systemInstruction"] = {"role": "system", "parts": system_parts}

    reply_text = await call_gemini_api(payload)

    return ChatRes(
        reply=reply_text,
        usage={"note": "Single-turn completion via /chat"}
    )

# ----------------------------
# Weather Query Endpoint
# ----------------------------
@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest):
    try:
        target_date = datetime.fromisoformat(payload.date).date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    loc_df = pick_location(df, payload.lat, payload.lon, payload.city)
    if loc_df.empty:
        raise HTTPException(status_code=404, detail="No data for the requested location.")

    target_doy = datetime(target_date.year, target_date.month, target_date.day).timetuple().tm_yday
    mask = doy_window_mask(loc_df["DOY"], target_doy, payload.window_days)
    win_df = loc_df[mask]

    if len(win_df) < 5:
        win_df = loc_df[loc_df["DATE"].dt.month == target_date.month]

    sample_n = len(win_df)
    if sample_n == 0:
        raise HTTPException(status_code=404, detail="No data found for the selected window/month.")

    city_name = (win_df["CITY"].mode().iat[0]
                 if "CITY" in win_df.columns and not win_df["CITY"].mode().empty else "Unknown")
    lat0 = float(win_df["LATITUDE"].mean())
    lon0 = float(win_df["LONGITUDE"].mean())

    metrics = {
        "mean_temp_c": round(safe_mean(win_df["T2M"]), 2),
        "mean_tmax_c": round(safe_mean(win_df["T2M_MAX"]), 2),
        "mean_tmin_c": round(safe_mean(win_df["T2M_MIN"]), 2),
        "mean_rh_pct": round(safe_mean(win_df["RH2M"]), 1),
        "mean_wind_ms": round(safe_mean(win_df["WS2M"]), 2)
    }

    probabilities = {
        "rain_probability_pct": round(pct_true(win_df["PRECTOTCORR"] >= payload.rain_mm_threshold), 1),
        "windy_probability_pct": round(pct_true(win_df["WS2M"] >= payload.windy_ms_threshold), 1),
        "hot_probability_pct": round(pct_true(win_df["T2M_MAX"] >= payload.hot_c_threshold), 1),
        "cold_probability_pct": round(pct_true(win_df["T2M_MIN"] <= payload.cold_c_threshold), 1)
    }

    friendly_advice = await get_ai_advice(
        city=city_name,
        metrics=metrics,
        probabilities=probabilities,
        target_date=payload.date
    )

    doy_lo = target_doy - payload.window_days
    doy_hi = target_doy + payload.window_days
    start_est = (datetime(target_date.year, 1, 1) + timedelta(days=max(0, doy_lo - 1))).date()
    end_est = (datetime(target_date.year, 1, 1) + timedelta(days=min(364, doy_hi - 1))).date()

    return QueryResponse(
        location={"city": city_name, "lat": round(lat0, 5), "lon": round(lon0, 5)},
        climatology_window={
            "start": start_est.isoformat(),
            "end": end_est.isoformat(),
            "window_days": str(payload.window_days)
        },
        sample_size=sample_n,
        metrics=metrics,
        probabilities=probabilities,
        friendly_advice=friendly_advice,
        debug={
            "thresholds": {
                "rain_mm_threshold": payload.rain_mm_threshold,
                "windy_ms_threshold": payload.windy_ms_threshold,
                "hot_c_threshold": payload.hot_c_threshold,
                "cold_c_threshold": payload.cold_c_threshold
            },
            "notes": "Advice generated by Gemini 2.5 Flash based on climatology data."
        }
    )

# ----------------------------
# Image Endpoint (disabled)
# ----------------------------
@app.post("/generate_image", response_model=ImageResponse)
async def generate_image(payload: ImageRequest):
    raise HTTPException(
        status_code=501,
        detail="Images API is not wired for this environment. Enable later with the correct endpoint/body."
    )

# ----------------------------
# Climatology Series Endpoint
# ----------------------------
@app.get("/climatology/series")
async def climatology_series(lat: Optional[float] = None,
                             lon: Optional[float] = None,
                             city: Optional[str] = None,
                             date: str = Query(..., description="YYYY-MM-DD"),
                             window_days: int = Query(7)):
    try:
        target_date = datetime.fromisoformat(date).date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    loc_df = pick_location(df, lat, lon, city)
    if loc_df.empty:
        raise HTTPException(status_code=404, detail="No data for the requested location.")

    target_doy = datetime(target_date.year, target_date.month, target_date.day).timetuple().tm_yday
    mask = doy_window_mask(loc_df["DOY"], target_doy, window_days)
    win_df = loc_df[mask].sort_values("DATE")
    if win_df.empty:
        raise HTTPException(status_code=404, detail="No data for the selected window.")

    return {
        "city": (win_df["CITY"].mode().iat[0] if "CITY" in win_df.columns and not win_df["CITY"].mode().empty else "Unknown"),
        "series": [
            {
                "date": d,
                "t2m": float(t),
                "tmax": float(tx),
                "tmin": float(tn),
                "rain_mm": float(p),
                "wind_ms": float(w)
            }
            for d, t, tx, tn, p, w in zip(
                win_df["DATE"].dt.strftime("%Y-%m-%d"),
                win_df["T2M"].values,
                win_df["T2M_MAX"].values,
                win_df["T2M_MIN"].values,
                win_df["PRECTOTCORR"].values,
                win_df["WS2M"].values
            )
        ]
    }