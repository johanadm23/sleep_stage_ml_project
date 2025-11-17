"""
src/serve.py

FastAPI app to serve the sleep-stage classifier.

Expecting model artifacts under ./models:
 - models/rf_model.joblib
 - models/label_encoder.joblib
 - models/meta.json   (contains {"features": [...]} )

Endpoints:
 - GET  /         -> health/status
 - POST /predict  -> accepts JSON {"features": [...]} and returns {"prediction": "...", "probs": [...], "classes": [...]}

Run locally (no Docker):
    pip install -r requirements-server.txt
    python src/serve.py

Run with uvicorn directly:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000
"""

import os
import json
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

# Optional: allow CORS for simple front-ends (uncomment if needed)
# from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = os.getenv("MODEL_PATH", "models/rf_model.joblib")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "models/label_encoder.joblib")
META_PATH = os.getenv("META_PATH", "models/meta.json")

# Request model: features must be a list of floats whose length equals model feature count.
# We use conlist with float to get basic validation; the length check below will provide more detailed errors.
class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: str
    probs: List[float]
    classes: List[str]

app = FastAPI(title="Sleep Stage Classifier", version="1.0")

# If you serve a frontend from another host, enable CORS (example below).
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Load model + metadata at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    _model_load_error = f"Could not load model from {MODEL_PATH}: {e}"

try:
    le = joblib.load(LABEL_ENCODER_PATH)
except Exception as e:
    le = None
    _le_load_error = f"Could not load label encoder from {LABEL_ENCODER_PATH}: {e}"

try:
    with open(META_PATH, "r") as f:
        meta = json.load(f)
        expected_features = meta.get("features", [])
except Exception as e:
    meta = None
    expected_features = []
    _meta_load_error = f"Could not load meta from {META_PATH}: {e}"

@app.get("/")
def read_root():
    ok = model is not None and le is not None and meta is not None
    details = {
        "model_loaded": model is not None,
        "label_encoder_loaded": le is not None,
        "meta_loaded": meta is not None,
        "feature_count_expected": len(expected_features),
    }
    # include errors if not loaded
    if not ok:
        details["errors"] = {
            k: v for k, v in {
                "model_error": globals().get("_model_load_error"),
                "le_error": globals().get("_le_load_error"),
                "meta_error": globals().get("_meta_load_error")
            }.items() if v is not None
        }
    return {"status": "ok" if ok else "error", "details": details}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None or le is None or meta is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded on server. Check / for details.")

    feats = req.features
    if len(expected_features) > 0 and len(feats) != len(expected_features):
        raise HTTPException(
            status_code=400,
            detail=f"Feature length mismatch: model expects {len(expected_features)} features {expected_features}, got {len(feats)}."
        )

    try:
        x = np.array(feats, dtype=float).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not convert features to numeric array: {e}")

    # predict_proba if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0].tolist()
        pred_idx = int(np.argmax(probs))
    else:
        # fallback to predict only
        pred_idx = int(model.predict(x)[0])
        probs = []

    # map label index back to string using label encoder if possible
    try:
        classes = le.inverse_transform(np.arange(len(le.classes_))).tolist()
        pred_label = le.inverse_transform([pred_idx])[0]
    except Exception:
        # fallback to meta classes if provided
        classes = meta.get("label_map", {}).values() if meta else []
        pred_label = str(pred_idx)

    return {"prediction": pred_label, "probs": probs, "classes": list(classes)}

# Allow running with `python src/serve.py` for easy local testing
if __name__ == "__main__":
    import uvicorn
    # The default port here is 8000; Dockerfile below runs on port 80.
    uvicorn.run("src.serve:app", host="0.0.0.0", port=8000, reload=False)
