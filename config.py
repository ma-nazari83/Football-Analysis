from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PLAYER_MODEL = MODEL_DIR / "yolo-football-player-detection.pt"
PITCH_MODEL = MODEL_DIR / "yolo-football-pitch-detection.pt"