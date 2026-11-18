
import os
import logging
from pathlib import Path
from time import perf_counter
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_ROOT = Path(
    os.getenv(
        "STUDENT_TEST_ROOT",
        "../ML_hackathon/Student folder/student_test"
    )
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("pathsnap")

app = FastAPI(
    title="PathSnap API",
    description="Distributed Map Matching and Route Reconstruction",
    version="1.0.0"
)

# Allow the UI on another laptop to communicate with the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this for final deployment if needed.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class PredictRequest(BaseModel):
    trip_id: str


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_trip_ids():

    if not DATA_ROOT.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Dataset directory not found: {DATA_ROOT}"
        )

    return sorted(
        p.name for p in DATA_ROOT.iterdir()
        if p.is_dir() and len(p.name) == 8 and p.name.isdigit()
    )


# --------------------------------------------------
# REQUEST LOGGING
# --------------------------------------------------

@app.middleware("http")
async def log_requests(request, call_next):

    start = perf_counter()

    response = await call_next(request)

    elapsed = perf_counter() - start

    logger.info(
        "%s %s | status=%s | time=%.4fs",
        request.method,
        request.url.path,
        response.status_code,
        elapsed
    )

    return response


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "message": "PathSnap API is running"
    }


# --------------------------------------------------
# LIST AVAILABLE TRIPS
# --------------------------------------------------

@app.get("/trips")
def list_trips():

    trips = get_trip_ids()

    return {
        "trips": trips
    }


# --------------------------------------------------
# PREDICTION ENDPOINT - TEMPORARY STUB
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictRequest):

    trip_ids = get_trip_ids()

    if request.trip_id not in trip_ids:
        raise HTTPException(
            status_code=404,
            detail="Trip not found"
        )

    logger.info("Prediction requested for trip %s", request.trip_id)

    # TEMPORARY STUB:
    # Real prediction logic will be added after the ML
    # model, candidate generation and HMM are integrated.

    return {
        "trip_id": request.trip_id,
        "status": "STUB - NOT A REAL PREDICTION",

        "predicted_route": [],
        "gps_points": [],

        "metrics": {
            "status": "Not evaluated yet"
        },

        "timings": {
            "total_time_s": None
        }
    }
