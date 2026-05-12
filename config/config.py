import os
from dotenv import load_dotenv

load_dotenv()

# ── Chemins — INCHANGE ───────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
HAM10000_DIR    = os.path.join(DATA_DIR, "HAM10000")
HAM10000_META   = os.path.join(HAM10000_DIR, "HAM10000_metadata.csv")
HAM10000_PART1  = os.path.join(HAM10000_DIR, "HAM10000_images_part_1")
HAM10000_PART2  = os.path.join(HAM10000_DIR, "HAM10000_images_part_2")
PH2_RAW_DIR     = os.path.join(DATA_DIR, "PH2")
PH2_CLEAN_DIR   = os.path.join(DATA_DIR, "PH2_clean")
PH2_IMG_DIR     = os.path.join(PH2_CLEAN_DIR, "images")
PH2_MASK_DIR    = os.path.join(PH2_CLEAN_DIR, "masks")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# ── Chemins — NOUVEAU Phase 2 ────────────────────────────────
LOGS_DIR            = os.path.join(BASE_DIR, "logs")
MLFLOW_DIR          = os.path.join(BASE_DIR, "mlruns")
CLASSIFIER_CKPT     = os.path.join(CHECKPOINTS_DIR, "classifier_best.pth")
SEGMENTOR_CKPT      = os.path.join(CHECKPOINTS_DIR, "segmentor_best.pth")

# ── Classes — INCHANGE ───────────────────────────────────────
LABEL_NAMES = {
    'nv': 'Naevi melanocytaires', 'mel': 'Melanome',
    'bkl': 'Keratose benigne', 'bcc': 'Carcinome basocellulaire',
    'akiec': 'Keratose actinique', 'vasc': 'Lesion vasculaire',
    'df': 'Dermatofibrome'
}

# ── Preprocessing — INCHANGE ─────────────────────────────────
IMAGE_SIZE      = (224, 224)
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

# ── Split — INCHANGE ─────────────────────────────────────────
TEST_SIZE       = 0.3
VAL_SIZE        = 0.5
RANDOM_STATE    = 42

# ── Entrainement de base — INCHANGE ──────────────────────────
BATCH_SIZE      = 8
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-4

# ── Entrainement — NOUVEAU Phase 2 ───────────────────────────
NUM_WORKERS         = None          # calcule automatiquement dans datasets.py
DROPOUT_P           = 0.5
NUM_CLASSES_HAM     = 7
NUM_CLASSES_PH2     = 1
FOCAL_GAMMA         = 2.0
LABEL_SMOOTHING     = 0.1
WEIGHT_DECAY        = 1e-4
LR_MIN              = 1e-6
WARMUP_EPOCHS       = 5
T_MAX               = NUM_EPOCHS
GRAD_CLIP_NORM      = 1.0
PATIENCE            = 10
DELTA               = 1e-4

# ── Kaggle — INCHANGE ────────────────────────────────────────
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY      = os.getenv("KAGGLE_KEY")

if not KAGGLE_USERNAME or not KAGGLE_KEY:
    raise EnvironmentError("Variables Kaggle manquantes dans .env")

# ── Création automatique des dossiers — NOUVEAU Phase 2 ──────
for directory in [LOGS_DIR, MLFLOW_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR]:
    os.makedirs(directory, exist_ok=True)