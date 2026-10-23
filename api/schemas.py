from pydantic import BaseModel, Field


class CorruptionRequest(BaseModel):
    corruption: str
    severity: int = Field(ge=1, le=5)
    seed: int | None = None


class PredictionResponse(BaseModel):
    class_id: int
    label: str
    confidence: float
    latency_ms: float
    probabilities: list[float]
    checkpoint: str
    trained: bool
