# config/config.py
# ============================================================
# Configuration centrale du projet
# Les valeurs sensibles sont lues depuis le fichier .env
# ============================================================

import os
from dotenv import load_dotenv

# Charger les variables depuis .env
load_dotenv()

# ── Chemins ──────────────────────────────────────────────────
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

# ── Classes HAM10000 ─────────────────────────────────────────
LABEL_NAMES = {
    'nv':    'Naevi melanocytaires',
    'mel':   'Melanome',
    'bkl':   'Keratose benigne',
    'bcc':   'Carcinome basocellulaire',
    'akiec': 'Keratose actinique',
    'vasc':  'Lesion vasculaire',
    'df':    'Dermatofibrome'
}

# ── Preprocessing ────────────────────────────────────────────
IMAGE_SIZE      = (224, 224)
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

# ── Split ────────────────────────────────────────────────────
TEST_SIZE       = 0.3
VAL_SIZE        = 0.5
RANDOM_STATE    = 42

# ── Entrainement ─────────────────────────────────────────────
BATCH_SIZE      = 8
NUM_WORKERS     = 2
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-4

# ── Kaggle API — lues depuis .env ────────────────────────────
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY      = os.getenv("KAGGLE_KEY")

# Verification que les variables sont bien chargees
if not KAGGLE_USERNAME or not KAGGLE_KEY:
    raise EnvironmentError(
        "Variables KAGGLE_USERNAME et KAGGLE_KEY manquantes.\n"
        "Creer un fichier .env a la racine du projet avec :\n"
        "  KAGGLE_USERNAME=ton_username\n"
        "  KAGGLE_KEY=ta_cle_api\n"
        "Voir .env.example pour le template."
    )
