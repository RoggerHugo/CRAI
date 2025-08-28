# app/schemas.py
from typing import Dict, Optional
from pydantic import BaseModel, Field

class ModelsOut(BaseModel):
    prediction: Dict[str, int] = Field(default_factory=dict)
    probabilities: Dict[str, float] = Field(default_factory=dict)

class PredictIn(BaseModel):
    student_id: Optional[str] = None
    features: Dict[str, float | int | str]
    request_id: Optional[str] = None
    use_model: Optional[str] = None  # "RandomForest" | "XGBoost" | "SVM"

class PredictOut(BaseModel):
    prediction_label: str
    score: Optional[float] = Field(default=None, ge=0, le=1)
    model_used: str
    model_version: str
    # NO Optional: siempre presente con dicts vacíos por defecto
    models: ModelsOut = Field(default_factory=ModelsOut)
