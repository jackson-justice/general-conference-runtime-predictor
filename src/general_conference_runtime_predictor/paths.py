from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_CACHE = ROOT / "data" / "cache"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

LEGACY_CSV = DATA_RAW / "talks.csv"
COLLECTED_CSV = DATA_PROCESSED / "talks_collected.csv"
DATASET_CSV = DATA_PROCESSED / "talks_dataset.csv"
BUNDLE_PATH = MODELS / "bundle.joblib"
PREDICTION_LOG = OUTPUTS / "predictions_log.csv"
