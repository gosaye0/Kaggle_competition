import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

def load_model(model_name = "final_pipeline_v1.pkl"):
    model_path = MODELS_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return joblib.load(model_path)