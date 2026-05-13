# PFA 2A — Detection et Segmentation de Lesions Cutanees

Projet de deep learning pour la **classification** (HAM10000) et la **segmentation** (PH2) de lesions cutanees dermoscopiques, developpe dans le cadre d'un projet de fin d'annee en 2eme annee.

---

## Table des matieres

1. [Apercu du projet](#apercu-du-projet)
2. [Architecture du projet](#architecture-du-projet)
3. [Prerequis](#prerequis)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Datasets](#datasets)
7. [Utilisation](#utilisation)
8. [Tests](#tests)
9. [Parametres](#parametres)
10. [Resultats attendus](#resultats-attendus)
11. [Problemes frequents](#problemes-frequents)
12. [Notes importantes](#notes-importantes)

---

## Apercu du projet

| Element | Detail |
|---|---|
| Tache 1 | Classification de 7 types de lesions — dataset HAM10000 (10 015 images) |
| Tache 2 | Segmentation binaire des lesions — dataset PH2 (200 images + masques) |
| Modele classification | ResNet50 fine-tune ImageNet |
| Modele segmentation | UNet encodeur ResNet34 pretraine |
| Loss classification | FocalLoss (gamma=2) + poids de classes |
| Loss segmentation | DiceBCELoss |
| Framework | PyTorch 2.0+ |
| Tracking | MLflow |

---

## Architecture du projet

```
pfa_Dectection_seg_LC_IM/
|
|-- config/
|   |-- __init__.py
|   `-- config.py                  # Parametres centralises, lit depuis .env
|
|-- data/
|   |-- download.py                # Telechargement automatique via Kaggle
|   `-- clean.py                   # Nettoyage PH2 + verification paires image/masque
|
|-- preprocessing/
|   |-- __init__.py
|   |-- transforms.py              # Augmentations albumentations HAM10000 + PH2
|   `-- datasets.py                # Dataset PyTorch + DataLoaders + WeightedSampler
|
|-- models/
|   |-- __init__.py
|   |-- classifier.py              # ResNet50 fine-tune
|   |-- segmentor.py               # UNet ResNet34
|   `-- losses.py                  # FocalLoss + DiceBCELoss + LabelSmoothingLoss
|
|-- utils/
|   |-- __init__.py
|   |-- seed.py                    # Reproductibilite totale
|   |-- metrics.py                 # F1, AUROC, Dice, IoU
|   |-- callbacks.py               # EarlyStopping + ModelCheckpoint
|   |-- tracking.py                # MLflow experiment tracking
|   `-- eda.py                     # Analyse exploratoire + visualisations
|
|-- tests/
|   |-- __init__.py
|   `-- test_datasets.py           # 34 tests unitaires
|
|-- outputs/                       # Graphiques EDA generes automatiquement
|-- checkpoints/                   # Poids des modeles sauvegardes
|-- logs/                          # Logs d'entrainement
|-- mlruns/                        # Runs MLflow
|-- main.py                        # Pipeline de donnees complet
|-- train.py                       # Boucle d'entrainement Phase 2
|-- evaluate.py                    # Evaluation finale sur test set
|-- predict.py                     # Inference sur nouvelles images
|-- conftest.py                    # Configuration pytest
|-- requirements.txt
|-- .env                           # Credentials Kaggle — NE PAS PUSHER
|-- .env.example                   # Template credentials pour collaborateurs
|-- .gitignore
`-- README.md
```

---

## Prerequis

| Outil | Version minimale | Lien |
|---|---|---|
| Python | 3.9+ | https://www.python.org |
| Git | derniere | https://git-scm.com |
| VS Code | derniere | https://code.visualstudio.com |

Extensions VS Code recommandees : Python, Jupyter, Pylance (Microsoft).

---

## Installation

### Etape 1 — Cloner le projet

```bash
git clone https://github.com/TON_USERNAME/PFA-2A-Skin-Lesion.git
cd PFA-2A-Skin-Lesion
```

### Etape 2 — Creer et activer l'environnement virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Etape 3 — Installer les dependances

```bash
pip install -r requirements.txt
```

### Verification de l'installation

```bash
python -c "
import torch, albumentations, torchmetrics, mlflow
print('PyTorch      :', torch.__version__)
print('Albumentations:', albumentations.__version__)
print('Torchmetrics :', torchmetrics.__version__)
print('MLflow       :', mlflow.__version__)
print('GPU          :', torch.cuda.is_available())
"
```

---

## Configuration

### Fichier .env — Credentials Kaggle

Copier le template et remplir les credentials :

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Contenu du fichier `.env` :

```
KAGGLE_USERNAME=ton_username_kaggle
KAGGLE_KEY=ta_cle_api_kaggle
```

Pour obtenir les credentials : https://www.kaggle.com/settings -> API -> Create New Token.

Verifier que la configuration est correcte :

```bash
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('USERNAME :', os.getenv('KAGGLE_USERNAME'))
print('KEY      :', os.getenv('KAGGLE_KEY')[:6] + '...')
"
```

---

## Datasets

### HAM10000 — Classification

| Classe | Description | Images | Pourcentage |
|---|---|---|---|
| nv | Naevi melanocytaires | 6705 | 66.9% |
| mel | Melanome | 1113 | 11.1% |
| bkl | Keratose benigne | 1099 | 11.0% |
| bcc | Carcinome basocellulaire | 514 | 5.1% |
| akiec | Keratose actinique | 327 | 3.3% |
| vasc | Lesion vasculaire | 142 | 1.4% |
| df | Dermatofibrome | 115 | 1.1% |

Dataset fortement desequilibre (ratio x58 entre nv et df). Les poids de classes et le `WeightedRandomSampler` sont appliques automatiquement.

### PH2 — Segmentation

200 images dermoscopiques avec masques annotes manuellement. Dimensions : ~765x575 pixels. Couverture moyenne de la lesion : 32.2% des pixels. Masques binaires : 0 = fond, 1 = lesion.

### Telechargement

```bash
python data/download.py
```

Taille totale : ~5.2 GB. Duree estimee : 10 a 30 minutes.

En cas de blocage de `unzip` sur Windows :

```bash
python -c "import zipfile; zipfile.ZipFile('NOM.zip').extractall('DOSSIER')"
```

---

## Utilisation

### Phase 1 — Pipeline de donnees

```bash
# 1. Nettoyer PH2 et verifier les paires image/masque
python data/clean.py

# 2. Analyse exploratoire — genere 4 graphiques dans outputs/
python utils/eda.py

# 3. Pipeline complet Phase 1
python main.py
```

### Phase 2 — Entrainement des modeles

```bash
# Lancer l'entrainement Classification + Segmentation
python train.py
```

Ce script execute dans l'ordre :
- Entrainement ResNet50 sur HAM10000 avec FocalLoss + EarlyStopping
- Entrainement UNet sur PH2 avec DiceBCELoss + EarlyStopping
- Sauvegarde des meilleurs poids dans `checkpoints/`
- Tracking complet dans MLflow

Pour visualiser les runs MLflow :

```bash
mlflow ui
# Ouvrir http://localhost:5000
```

### Evaluation

```bash
python evaluate.py
```

Genere dans `outputs/` :
- Matrice de confusion HAM10000
- Exemples de segmentation PH2 (image / masque reel / masque predit)

### Inference sur une image

```bash
# Classification + Segmentation
python predict.py --image path/to/image.jpg --task both

# Classification uniquement
python predict.py --image path/to/image.jpg --task classification

# Segmentation uniquement
python predict.py --image path/to/image.jpg --task segmentation
```

### Ordre d'execution complet

```bash
# Activer le venv
venv\Scripts\activate

# Phase 1 — Donnees
python data/download.py
python data/clean.py
python utils/eda.py

# Phase 2 — Entrainement
python train.py

# Evaluation
python evaluate.py
```

---

## Tests

```bash
pytest tests/test_datasets.py -v
```

La suite couvre 34 tests :

- Shapes des batches images et labels/masques
- Valeurs normalisees dans [-3, 3]
- Masques strictement binaires {0, 1}
- Absence de data leakage entre train / val / test
- Proportions du split 70/15/15
- Alignement image masque PH2
- Shapes de sortie ResNet50 et UNet
- Softmax qui somme a 1
- Comportement correct FocalLoss et DiceBCELoss

Resultat attendu :

```
34 passed
```

---

## Parametres

Tous les parametres sont centralises dans `config/config.py`.

```python
# Preprocessing
IMAGE_SIZE      = (224, 224)
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

# Split
TEST_SIZE       = 0.3        # 30% pour val + test
VAL_SIZE        = 0.5        # 50% du temp = 15% total
RANDOM_STATE    = 42         # Seed reproductibilite

# Modeles
DROPOUT_P       = 0.5        # Regularisation Dropout
NUM_CLASSES_HAM = 7
NUM_CLASSES_PH2 = 1

# Losses
FOCAL_GAMMA     = 2.0        # Focusing parameter FocalLoss
LABEL_SMOOTHING = 0.1        # Evite la sur-confiance

# Entrainement
BATCH_SIZE      = 8          # Reduire a 4 si erreur memoire GPU
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4       # Regularisation L2
LR_MIN          = 1e-6       # Learning rate minimum CosineAnnealing
WARMUP_EPOCHS   = 5          # Stabilisation debut entrainement
GRAD_CLIP_NORM  = 1.0        # Evite explosion du gradient
PATIENCE        = 10         # EarlyStopping
```

---

## Resultats attendus

| Metrique | Modele | Objectif |
|---|---|---|
| F1 Macro | ResNet50 | > 0.75 |
| AUROC | ResNet50 | > 0.90 |
| Dice Score | UNet | > 0.85 |
| IoU | UNet | > 0.75 |

---

## Problemes frequents

**EnvironmentError : Variables Kaggle manquantes**
```bash
copy .env.example .env    # Windows
cp .env.example .env      # Mac/Linux
# Remplir les credentials dans .env
```

**ModuleNotFoundError**
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

**CUDA out of memory**
```python
# Reduire dans config/config.py
BATCH_SIZE = 4
```

**NUM_WORKERS erreur sur Windows**
```python
# Mettre dans config/config.py
NUM_WORKERS = 0
```

**unzip bloque sur Windows**
```bash
python -c "import zipfile; zipfile.ZipFile('NOM.zip').extractall('DOSSIER')"
```

**Session Colab reinitalisee**
```python
from google.colab import drive
drive.mount('/content/drive')
# Modifier DATA_DIR dans config/config.py pour pointer vers Drive
```

---

## Notes importantes

- Ne jamais versionner `data/` sur GitHub — trop lourd, presente dans `.gitignore`
- Ne jamais versionner `.env` — credentials Kaggle, presente dans `.gitignore`
- Toujours activer le venv avant de lancer une commande Python
- Travailler sur la branche `develop`, jamais directement sur `main`
- `conftest.py` est necessaire a la racine pour que pytest trouve les modules
- Modifier uniquement `config/config.py` pour changer les parametres globaux
- Les graphiques EDA sont dans `outputs/`, les poids des modeles dans `checkpoints/`