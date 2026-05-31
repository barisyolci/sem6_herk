import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "output"

DEFAULT_TARGET_FILE = ASSETS_DIR / "target_role.txt"
DEFAULT_LABELS_FILE = ASSETS_DIR / "ground_truth.csv"
DEFAULT_RESUMES_DIR = ASSETS_DIR / "resumes"

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)