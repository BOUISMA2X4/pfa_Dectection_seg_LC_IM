# PFA 2A — Détection et Segmentation de Lésions Cutanées

Projet de deep learning pour la **classification** (HAM10000) et la **segmentation** (PH2) de lésions cutanées dermoscopiques, développé dans le cadre d'un projet de fin d'année en 2ème année.

---

## Table des matières

1. [Aperçu du projet](#aperçu-du-projet)
2. [Architecture du projet](#architecture-du-projet)
3. [Prérequis](#prérequis)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Datasets](#datasets)
7. [Utilisation](#utilisation)
8. [Tests](#tests)
9. [Paramètres](#paramètres)
10. [Résultats attendus](#résultats-attendus)
11. [Problèmes fréquents](#problèmes-fréquents)
12. [Notes importantes](#notes-importantes)

---

## Aperçu du projet

| Élément | Détail |
|---|---|
| Tâche 1 | Classification de 7 types de lésions — dataset HAM10000 (10 015 images) |
| Tâche 2 | Segmentation binaire des lésions — dataset PH2 (200 images + masques) |
| Modèle classification | ResNet50 fine-tuné ImageNet |
| Modèle segmentation | UNet encodeur ResNet34 pré-entraîné |
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
|   `-- config.py                  # Paramètres centralisés, lit depuis .env
|
|-- data/
|   |-- download.py                # Téléchargement automatique via Kaggle
|   `-- clean.py                   # Nettoyage PH2 + vérification paires image/masque
|
|-- preprocessing/
|   |-- __init__.py
|   |-- transforms.py              # Augmentations albumentations HAM10000 + PH2
|   `-- datasets.py                # Dataset PyTorch + DataLoaders + WeightedSampler
|
|-- models/
|   |-- __init__.py
|   |-- classifier.py              # ResNet50 fine-tuné
|   |-- segmentor.py               # UNet ResNet34
|   `-- losses.py                  # FocalLoss + DiceBCELoss + LabelSmoothingLoss
|
|-- utils/
|   |-- __init__.py
|   |-- seed.py                    # Reproductibilité totale
|   |-- metrics.py                 # F1, AUROC, Dice, IoU
|   |-- callbacks.py               # EarlyStopping + ModelCheckpoint
|   |-- tracking.py                # MLflow experiment tracking
|   `-- eda.py                     # Analyse exploratoire + visualisations
|
|-- tests/
|   |-- __init__.py
|   `-- test_datasets.py           # 34 tests unitaires
|
|-- outputs/                       # Graphiques EDA générés automatiquement
|-- checkpoints/                   # Poids des modèles sauvegardés
|-- logs/                          # Logs d'entraînement
|-- mlruns/                        # Runs MLflow
|-- main.py                        # Pipeline de données complet
|-- train.py                       # Boucle d'entraînement Phase 2
|-- evaluate.py                    # Évaluation finale sur test set
|-- predict.py                     # Inférence sur nouvelles images
|-- conftest.py                    # Configuration pytest
|-- requirements.txt
|-- .env                           # Credentials Kaggle — NE PAS POUSSER
|-- .env.example                   # Template credentials pour collaborateurs
|-- .gitignore
`-- README.md
```

---

## Prérequis

| Outil | Version minimale | Lien |
|---|---|---|
| Python | 3.9+ | https://www.python.org |
| Git | dernière | https://git-scm.com |
| VS Code | dernière | https://code.visualstudio.com |

Extensions VS Code recommandées : Python, Jupyter, Pylance (Microsoft).

---

## Installation

### Étape 1 — Cloner le projet

```bash
git clone https://github.com/BOUISMA2X4/pfa_Dectection_seg_LC_IM.git
cd pfa_Dectection_seg_LC_IM
```

### Étape 2 — Créer et activer l'environnement virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Étape 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### Vérification de l'installation

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

Pour obtenir les credentials : https://www.kaggle.com/settings → API → Create New Token.

Vérifier que la configuration est correcte :

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
| nv | Nævi mélanocytaires | 6705 | 66.9% |
| mel | Mélanome | 1113 | 11.1% |
| bkl | Kératose bénigne | 1099 | 11.0% |
| bcc | Carcinome basocellulaire | 514 | 5.1% |
| akiec | Kératose actinique | 327 | 3.3% |
| vasc | Lésion vasculaire | 142 | 1.4% |
| df | Dermatofibrome | 115 | 1.1% |

Dataset fortement déséquilibré (ratio x58 entre `nv` et `df`). Les poids de classes et le `WeightedRandomSampler` sont appliqués automatiquement.

### PH2 — Segmentation

200 images dermoscopiques avec masques annotés manuellement. Dimensions : ~765×575 pixels. Couverture moyenne de la lésion : 32.2% des pixels. Masques binaires : 0 = fond, 1 = lésion.

### Téléchargement

```bash
python data/download.py
```

Taille totale : ~5.2 Go. Durée estimée : 10 à 30 minutes.

En cas de blocage de `unzip` sur Windows :

```bash
python -c "import zipfile; zipfile.ZipFile('NOM.zip').extractall('DOSSIER')"
```

---

## Utilisation

### Phase 1 — Pipeline de données

```bash
# 1. Nettoyer PH2 et vérifier les paires image/masque
python data/clean.py

# 2. Analyse exploratoire — génère 4 graphiques dans outputs/
python utils/eda.py

# 3. Pipeline complet Phase 1
python main.py
```

### Phase 2 — Entraînement des modèles

```bash
python train.py
```

Ce script exécute dans l'ordre :
- Entraînement ResNet50 sur HAM10000 avec FocalLoss + EarlyStopping
- Entraînement UNet sur PH2 avec DiceBCELoss + EarlyStopping
- Sauvegarde des meilleurs poids dans `checkpoints/`
- Tracking complet dans MLflow

Pour visualiser les runs MLflow :

```bash
mlflow ui
# Ouvrir http://localhost:5000
```

### Évaluation

```bash
python evaluate.py
```

Génère dans `outputs/` :
- Matrice de confusion HAM10000
- Exemples de segmentation PH2 (image / masque réel / masque prédit)

### Inférence sur une image

```bash
# Classification + Segmentation
python predict.py --image path/to/image.jpg --task both

# Classification uniquement
python predict.py --image path/to/image.jpg --task classification

# Segmentation uniquement
python predict.py --image path/to/image.jpg --task segmentation
```

### Ordre d'exécution complet

```bash
# Activer le venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# Phase 1 — Données
python data/download.py
python data/clean.py
python utils/eda.py

# Phase 2 — Entraînement
python train.py

# Évaluation
python evaluate.py
```

---

## Tests

```bash
pytest tests/test_datasets.py -v
```

La suite couvre 34 tests :

- Shapes des batches images et labels/masques
- Valeurs normalisées dans [-3, 3]
- Masques strictement binaires {0, 1}
- Absence de data leakage entre train / val / test
- Proportions du split 70/15/15
- Alignement image/masque PH2
- Shapes de sortie ResNet50 et UNet
- Softmax qui somme à 1
- Comportement correct FocalLoss et DiceBCELoss

Résultat attendu :

```
34 passed
```

---

## Paramètres

Tous les paramètres sont centralisés dans `config/config.py`.

```python
# Preprocessing
IMAGE_SIZE      = (224, 224)
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

# Split
TEST_SIZE       = 0.3        # 30% pour val + test
VAL_SIZE        = 0.5        # 50% du reste = 15% total
RANDOM_STATE    = 42         # Seed reproductibilité

# Modèles
DROPOUT_P       = 0.5        # Régularisation Dropout
NUM_CLASSES_HAM = 7
NUM_CLASSES_PH2 = 1

# Losses
FOCAL_GAMMA     = 2.0        # Focusing parameter FocalLoss
LABEL_SMOOTHING = 0.1        # Évite la sur-confiance

# Entraînement
BATCH_SIZE      = 8          # Réduire à 4 si erreur mémoire GPU
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4       # Régularisation L2
LR_MIN          = 1e-6       # Learning rate minimum CosineAnnealing
WARMUP_EPOCHS   = 5          # Stabilisation début entraînement
GRAD_CLIP_NORM  = 1.0        # Évite explosion du gradient
PATIENCE        = 10         # EarlyStopping
```

---

## Résultats attendus

| Métrique | Modèle | Objectif |
|---|---|---|
| F1 Macro | ResNet50 | > 0.75 |
| AUROC | ResNet50 | > 0.90 |
| Dice Score | UNet | > 0.85 |
| IoU | UNet | > 0.75 |

---

## Problèmes fréquents

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
# Réduire dans config/config.py
BATCH_SIZE = 4
```

**NUM_WORKERS erreur sur Windows**
```python
# Mettre dans config/config.py
NUM_WORKERS = 0
```

**unzip bloqué sur Windows**
```bash
python -c "import zipfile; zipfile.ZipFile('NOM.zip').extractall('DOSSIER')"
```

**Session Colab réinitialisée**
```python
from google.colab import drive
drive.mount('/content/drive')
# Modifier DATA_DIR dans config/config.py pour pointer vers Drive
```

---

## Notes importantes

- Ne jamais versionner `data/` sur GitHub — trop lourd, présent dans `.gitignore`
- Ne jamais versionner `.env` — credentials Kaggle, présent dans `.gitignore`
- Toujours activer le venv avant de lancer une commande Python
- Travailler sur la branche `develop`, jamais directement sur `main`
- `conftest.py` est nécessaire à la racine pour que pytest trouve les modules
- Modifier uniquement `config/config.py` pour changer les paramètres globaux
- Les graphiques EDA sont dans `outputs/`, les poids des modèles dans `checkpoints/`