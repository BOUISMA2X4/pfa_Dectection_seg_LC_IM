# config/config.py
import os

# ── Chemins ──────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR         = os.path.join(BASE_DIR, "data")

HAM10000_DIR     = os.path.join(DATA_DIR, "HAM10000")
HAM10000_META    = os.path.join(HAM10000_DIR, "HAM10000_metadata.csv")
HAM10000_PART1   = os.path.join(HAM10000_DIR, "HAM10000_images_part_1")
HAM10000_PART2   = os.path.join(HAM10000_DIR, "HAM10000_images_part_2")

PH2_RAW_DIR      = os.path.join(DATA_DIR, "PH2")
PH2_CLEAN_DIR    = os.path.join(DATA_DIR, "PH2_clean")
PH2_IMG_DIR      = os.path.join(PH2_CLEAN_DIR, "images")
PH2_MASK_DIR     = os.path.join(PH2_CLEAN_DIR, "masks")

OUTPUTS_DIR      = os.path.join(BASE_DIR, "outputs")
CHECKPOINTS_DIR  = os.path.join(BASE_DIR, "checkpoints")

# ── Classes HAM10000 ─────────────────────────────────────────
LABEL_NAMES = {
    'nv':    'Nævi mélanocytaires',
    'mel':   'Mélanome',
    'bkl':   'Kératose bénigne',
    'bcc':   'Carcinome basocellulaire',
    'akiec': 'Kératose actinique',
    'vasc':  'Lésion vasculaire',
    'df':    'Dermatofibrome'
}

# ── Preprocessing ────────────────────────────────────────────
IMAGE_SIZE       = (224, 224)
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]

# ── Split ────────────────────────────────────────────────────
TEST_SIZE        = 0.3
VAL_SIZE         = 0.5
RANDOM_STATE     = 42

# ── Entraînement ─────────────────────────────────────────────
BATCH_SIZE       = 8
NUM_WORKERS      = 2
NUM_EPOCHS       = 30
LEARNING_RATE    = 1e-4

# ── Kaggle API ───────────────────────────────────────────────
KAGGLE_USERNAME  = "med sdt"
KAGGLE_KEY       = "KGAT_f02baeb30a5f223df64e9ccb6d2d4de9"