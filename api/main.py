from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from robustvision.config import get_settings
from robustvision.data.corruptions import SEVERITY, apply_corruption
from robustvision.device import device_metadata, select_device
from robustvision.models.inference import predict_image
from robustvision.models.registry import model_metadata
from robustvision.services import available_checkpoints, get_checkpoint

from .schemas import PredictionResponse

settings = get_settings()
app = FastAPI(title="RobustVision AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _image(data: bytes) -> Image.Image:
    if not data:
        raise HTTPException(422, "Empty upload")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "Upload exceeds configured limit")
    try:
        with Image.open(BytesIO(data)) as source:
            return source.convert("RGB").copy()
    except UnidentifiedImageError as exc:
        raise HTTPException(415, "Upload must be a valid image") from exc


@app.get("/health")
def health():
    return {"status": "ok", "device": device_metadata(settings.device)}


@app.get("/models")
def models():
    return [model_metadata("baseline_cnn"), model_metadata("resnet18")]


@app.get("/checkpoints")
def checkpoints():
    return available_checkpoints()


@app.get("/corruptions")
def corruptions():
    return SEVERITY


@app.post("/predict/{checkpoint_id}", response_model=PredictionResponse)
async def predict(checkpoint_id: str, file: UploadFile = File(...)):
    image = _image(await file.read())
    try:
        device = select_device(settings.device)
        model, metadata = get_checkpoint(checkpoint_id, device)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return {**predict_image(model, image, device), "checkpoint": checkpoint_id, "trained": True}


@app.post("/preview-corruption/{corruption}")
async def preview(
    corruption: str, severity: int, file: UploadFile = File(...), seed: int | None = None
):
    try:
        result = apply_corruption(_image(await file.read()), corruption, severity, seed)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    out = BytesIO()
    result.save(out, "PNG")
    from fastapi.responses import Response

    return Response(out.getvalue(), media_type="image/png")
