"""FastAPI Application for PathSnap Distributed Map Matching.

Runs on Laptop 1 during evaluation.
Accepts trip evaluation requests from Laptop 2 UI, runs end-to-end inference,
and returns prediction metrics and route geometry.
"""
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..data.loader import list_trips
from ..models.pipeline import MapMatchingPipeline, DEFAULT_DATA_ROOT, DEFAULT_MODEL_PATH

# Configure formatted console logging for evaluator inspection
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PathSnapAPI")

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed candidate classifier + HMM/Viterbi map matching service.",
    version="2.0.0"
)

# Enable CORS for cross-laptop requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy loaded)
pipeline: Optional[MapMatchingPipeline] = None


def get_pipeline() -> MapMatchingPipeline:
    global pipeline
    if pipeline is None:
        logger.info(f"Initializing MapMatchingPipeline with model at {DEFAULT_MODEL_PATH}...")
        pipeline = MapMatchingPipeline()
        logger.info(f"Pipeline ready. Active model: {pipeline.model_name}")
    return pipeline


@app.get("/api/health")
def health_check():
    pipe = get_pipeline()
    return {
        "status": "healthy",
        "system": "PathSnap Map-Matching Engine",
        "model": pipe.model_name,
        "dataset_root": str(pipe.data_root)
    }


@app.get("/api/trips")
def get_evaluation_trips(split: str = Query("evaluator_test", description="Dataset split (evaluator_test, student_test, train)")):
    """List available trips for evaluation."""
    try:
        trips = list_trips(str(DEFAULT_DATA_ROOT), split)
        return {
            "split": split,
            "count": len(trips),
            "trips": trips
        }
    except Exception as e:
        logger.error(f"Error listing trips for split {split}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate/{trip_id}")
@app.get("/api/evaluate/{trip_id}")
def evaluate_trip(
    trip_id: str,
    split: str = Query("evaluator_test", description="Dataset partition"),
    radius: float = Query(50.0, description="Candidate search radius in metres")
):
    """Execute complete end-to-end map matching on a selected trip."""
    pipe = get_pipeline()

    print("\n" + "=" * 65)
    print(f"🚀 [INFERENCE REQUEST] Received evaluation for Trip: {trip_id} (Split: {split})")
    print("=" * 65)

    t0 = time.time()
    try:
        result = pipe.process_trip(trip_id=trip_id, split=split, radius=radius)
    except Exception as e:
        logger.error(f"Error processing trip {trip_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    duration_ms = (time.time() - t0) * 1000.0
    metrics = result.get("metrics", {})
    acc_info = metrics.get("accuracy", {})
    seq_info = metrics.get("sequence", {})

    print(f"✅ [COMPLETED] Trip {trip_id} processed in {duration_ms:.1f}ms")
    print(f"   • HMM Pointwise Accuracy:       {acc_info.get('hmm_pointwise_accuracy', 'N/A')}%")
    print(f"   • Classifier-Only Accuracy:     {acc_info.get('classifier_pointwise_accuracy', 'N/A')}%")
    print(f"   • Net HMM Accuracy Gain:        +{acc_info.get('hmm_accuracy_gain', 'N/A')}%")
    print(f"   • Route Continuity:             {seq_info.get('continuity', 'N/A')}%")
    print(f"   • Sequence F1 Score:            {seq_info.get('f1_score', 'N/A')}%")
    print(f"   • Total Points Processed:       {metrics.get('n_gps_points')}")
    print("=" * 65 + "\n")

    return result


@app.get("/api/map/{trip_id}", response_class=HTMLResponse)
def get_trip_folium_map(
    trip_id: str,
    split: str = Query("evaluator_test", description="Dataset partition")
):
    """Generate an interactive Folium map visualization."""
    import folium
    pipe = get_pipeline()
    result = pipe.process_trip(trip_id=trip_id, split=split)
    geo = result["geo"]

    m = folium.Map(location=geo["center"], zoom_start=15, tiles="CartoDB positron")

    # Layer: Raw GPS Track
    gps_layer = folium.FeatureGroup(name="Raw GPS Track (Observations)", show=True)
    for lat, lon in geo["gps_track"]:
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#3b82f6",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.6,
            weight=1
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    # Layer: Predicted Route
    pred_layer = folium.FeatureGroup(name="PathSnap Predicted Route (HMM)", show=True)
    for seg in geo["predicted_route"]:
        folium.PolyLine(
            locations=seg,
            color="#10b981",
            weight=5,
            opacity=0.85
        ).add_to(pred_layer)
    pred_layer.add_to(m)

    # Layer: Reference Route (Ground Truth)
    if geo["reference_route"]:
        ref_layer = folium.FeatureGroup(name="Reference Ground Truth (.route)", show=True)
        for seg in geo["reference_route"]:
            folium.PolyLine(
                locations=seg,
                color="#f59e0b",
                weight=3,
                opacity=0.9,
                dash_array="6, 6"
            ).add_to(ref_layer)
        ref_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render()


# Serve UI static files if directory exists
ui_dir = Path(__file__).resolve().parents[2] / "ui"
if ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    print("\nStarting PathSnap Distributed Map Matching API on 0.0.0.0:8000...")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
