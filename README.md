# PFA 2A - Detection et Segmentation de Lesions Cutanees

Projet de deep learning pour la classification (HAM10000) et la segmentation (PH2)
de lesions cutanees dermoscopiques.

---

## Table des matieres

1. [Prerequis](#prerequis)
2. [Architecture du projet](#architecture-du-projet)
3. [Installation](#installation)
4. [Configuration des variables d'environnement](#configuration-des-variables-denvironnement)
5. [Configuration Kaggle](#configuration-kaggle)
6. [Telechargement des datasets](#telechargement-des-datasets)
7. [Nettoyage des donnees](#nettoyage-des-donnees)
8. [Analyse exploratoire EDA](#analyse-exploratoire-eda)
9. [Pipeline complet](#pipeline-complet)
10. [Verification PyTorch](#verification-pytorch)
11. [Parametres configurables](#parametres-configurables)
12. [Problemes frequents](#problemes-frequents)

---

## Prerequis

Avant de commencer, installer les outils suivants :

| Outil    | Version minimale | Lien                          |
|----------|-----------------|-------------------------------|
| Python   | 3.9+            | https://www.python.org        |
| Git      | derniere        | https://git-scm.com           |
| VS Code  | derniere        | https://code.visualstudio.com |

Extensions VS Code recommandees :
- Python (Microsoft)
- Jupyter (Microsoft)
- Pylance (Microsoft)

---

## Architecture du projet

```
pfa_Dectection_seg_LC_IM/
|
|-- config/
|   `-- config.py              # Tous les parametres centralises (lit depuis .env)
|
|-- data/
|   |-- download.py            # Telechargement des datasets via Kaggle
|   `-- clean.py               # Nettoyage PH2 + verification paires
|
|-- preprocessing/
|   |-- transforms.py          # Augmentations HAM10000 et PH2 synchronisees
|   `-- datasets.py            # Classes Dataset PyTorch + DataLoaders
|
|-- utils/
|   `-- eda.py                 # Analyse exploratoire + visualisations
|
|-- outputs/                   # Graphiques generes automatiquement
|-- checkpoints/               # Poids des modeles sauvegardes
|-- main.py                    # Point d'entree principal
|-- requirements.txt           # Dependances Python
|-- .env                       # Variables d'environnement - NE PAS PUSHER
|-- .env.example               # Template .env pour les collaborateurs
|-- .gitignore                 # Fichiers exclus de Git
`-- README.md                  # Ce fichier
```

---

## Installation

### Etape 1 - Cloner le projet

Utiliser cette commande pour recuperer le projet depuis GitHub.
A faire une seule fois sur chaque machine.

```bash
git clone https://github.com/TON_USERNAME/PFA-2A-Skin-Lesion.git
cd PFA-2A-Skin-Lesion
```

### Etape 2 - Creer l'environnement virtuel

L'environnement virtuel isole les dependances du projet.
A faire une seule fois par machine.

```bash
python -m venv venv
```

### Etape 3 - Activer l'environnement virtuel

A faire a chaque fois que vous ouvrez un nouveau terminal.

```bash
# Sur Windows :
venv\Scripts\activate

# Sur Mac / Linux :
source venv/bin/activate
```

Verification : le prefixe (venv) doit apparaitre dans le terminal.

```
(venv) C:\Users\...>
```

### Etape 4 - Installer les dependances

Installe tous les packages necessaires listes dans requirements.txt.
A faire une seule fois apres avoir active le venv.

```bash
pip install -r requirements.txt
```

Duree estimee : 5 a 15 minutes selon la connexion (PyTorch fait environ 2 GB).

### Verification de l'installation

```bash
python -c "import torch; import sklearn; import pandas; import matplotlib; print('Installation OK')"
```

Resultat attendu :
```
Installation OK
```

---

## Configuration des variables d'environnement

### Pourquoi un fichier .env ?

Les credentials Kaggle ne doivent jamais etre ecrits directement dans le code
et ne doivent jamais etre pousses sur GitHub.
Le fichier .env stocke ces informations de facon securisee sur ta machine uniquement.

### Etape 5 - Creer le fichier .env

Un fichier template .env.example est fourni dans le projet.
Il suffit de le copier et de le remplir.

```bash
# Sur Windows :
copy .env.example .env

# Sur Mac / Linux :
cp .env.example .env
```

### Etape 6 - Remplir le fichier .env

Ouvrir le fichier .env et remplacer les valeurs par tes credentials Kaggle :

```
KAGGLE_USERNAME=ton_username_kaggle
KAGGLE_KEY=ta_cle_api_kaggle
```

Pour recuperer tes credentials Kaggle :
1. Aller sur https://www.kaggle.com/settings
2. Section API
3. Cliquer sur "Create New Token"
4. Un fichier kaggle.json est telecharge contenant username et key

### Verifier que le fichier .env est bien charge

```bash
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
username = os.getenv('KAGGLE_USERNAME')
key      = os.getenv('KAGGLE_KEY')
print('KAGGLE_USERNAME :', username if username else 'MANQUANT')
print('KAGGLE_KEY      :', key[:6] + '...' if key else 'MANQUANT')
"
```

Resultat attendu :
```
KAGGLE_USERNAME : med sdt
KAGGLE_KEY      : KGAT_f...
```

### Fichiers .env et securite

| Fichier       | Sur GitHub | Role                                      |
|---------------|------------|-------------------------------------------|
| .env          | Non        | Contient tes vraies cles API              |
| .env.example  | Oui        | Template vide a remplir par le binome     |

Le fichier .env est protege par .gitignore et ne sera jamais pousse sur GitHub.

---

## Configuration Kaggle

### Pourquoi Kaggle ?

Les trois datasets du projet sont heberges sur Kaggle.
L'API Kaggle permet de les telecharger directement en ligne de commande.

### Etape 7 - Installer Kaggle

```bash
pip install kaggle
```

### Verifier la configuration

Les credentials sont lus automatiquement depuis le fichier .env via config/config.py.
Pour tester que la connexion fonctionne :

```bash
kaggle datasets list
```

Si cette commande retourne une liste de datasets, la configuration est correcte.

---

## Telechargement des datasets

### Etape 8 - Lancer le telechargement automatique

Ce script configure Kaggle et telecharge les 3 datasets en une seule commande.
A lancer une seule fois depuis la racine du projet.

```bash
python data/download.py
```

Ce qui est telecharge :

| Dataset   | Taille   | Contenu                                        |
|-----------|----------|------------------------------------------------|
| HAM10000  | 3.5 GB   | 10 015 images .jpg + HAM10000_metadata.csv     |
| DermNet   | 1.5 GB   | Images de maladies de peau                     |
| PH2       | 200 MB   | 200 images .bmp + 200 masques .bmp annotes     |

Duree estimee : 10 a 30 minutes selon la connexion.

### Telechargement manuel (si le script echoue)

En cas d'echec du script, telecharger chaque dataset individuellement :

```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/HAM10000 --unzip
kaggle datasets download -d shubhamgoel27/dermnet -p data/DermNet --unzip
kaggle datasets download -d spacesurfer/ph2-dataset -p data/PH2 --unzip
```

### Extraction manuelle sur Windows (si unzip bloque)

Sur Windows, la commande unzip peut se bloquer indefiniment.
Utiliser Python a la place :

```bash
python -c "import zipfile; zipfile.ZipFile('skin-cancer-mnist-ham10000.zip').extractall('data/HAM10000'); print('HAM10000 OK')"
python -c "import zipfile; zipfile.ZipFile('dermnet.zip').extractall('data/DermNet'); print('DermNet OK')"
python -c "import zipfile; zipfile.ZipFile('ph2-dataset.zip').extractall('data/PH2'); print('PH2 OK')"
```

### Verifier que les datasets sont bien telecharges

```bash
python -c "
import os
datasets = {'HAM10000': 'data/HAM10000', 'DermNet': 'data/DermNet', 'PH2': 'data/PH2'}
for name, path in datasets.items():
    if os.path.exists(path):
        count = sum(len(f) for _, _, f in os.walk(path))
        print(f'[OK] {name} : {count} fichiers')
    else:
        print(f'[MANQUANT] {name} : dossier introuvable')
"
```

---

## Nettoyage des donnees

### Quand utiliser cette commande ?

A lancer une seule fois apres le telechargement de PH2.
PH2 a une structure imbriquee complexe. Ce script la reorganise en deux dossiers plats.

### Etape 9 - Nettoyer PH2

```bash
python data/clean.py
```

Ce que fait ce script dans l'ordre :

1. Parcourt la structure complexe de PH2 (un sous-dossier par image)
2. Copie chaque image vers data/PH2_clean/images/
3. Copie chaque masque vers data/PH2_clean/masks/
4. Binarise les masques : pixel > 127 devient 1 (lesion), sinon 0 (fond)
5. Verifie que chaque image a exactement son masque correspondant

Structure creee apres nettoyage :

```
data/PH2_clean/
|-- images/
|   |-- IMD002.bmp
|   |-- IMD003.bmp
|   `-- ...  (200 fichiers au total)
`-- masks/
    |-- IMD002.bmp
    |-- IMD003.bmp
    `-- ...  (200 fichiers au total)
```

Resultat attendu dans le terminal :

```
[OK] PH2 nettoye - 200 paires image/masque copiees
[OK] 200 masques binarises
[OK] 200 paires image/masque verifiees - aucun mismatch
```

### Verifier le nettoyage manuellement

```bash
python -c "
import os
imgs  = len(os.listdir('data/PH2_clean/images'))
masks = len(os.listdir('data/PH2_clean/masks'))
print(f'Images  : {imgs}')
print(f'Masques : {masks}')
print('OK' if imgs == masks == 200 else 'ERREUR - mismatch detecte')
"
```

### Verifier la correspondance HAM10000

```bash
python -c "
import pandas as pd, os
meta = pd.read_csv('data/HAM10000/HAM10000_metadata.csv')
found = 0
for iid in meta['image_id']:
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        if os.path.exists(f'data/HAM10000/{part}/{iid}.jpg'):
            found += 1
            break
print(f'Images trouvees : {found} / {len(meta)}')
"
```

---

## Analyse exploratoire EDA

### Quand utiliser cette commande ?

A lancer apres le nettoyage pour visualiser les donnees avant l'entrainement.
Genere des graphiques utiles pour le rapport et pour comprendre les donnees.

### Etape 10 - Lancer l'EDA

```bash
python utils/eda.py
```

Duree estimee : 5 a 10 minutes (parcourt les 10 000 images HAM10000).

Fichiers generes dans outputs/ :

| Fichier                 | Contenu                                                  |
|-------------------------|----------------------------------------------------------|
| class_distribution.png  | 4 graphiques sur le desequilibre des 7 classes           |
| representative_grid.png | Grille 7 lignes x 5 colonnes d'images par categorie      |
| ham_image_quality.png   | Dimensions + luminosite + luminosite par classe          |
| ph2_image_quality.png   | Dimensions + luminosite + couverture masque PH2          |

---

## Pipeline complet

### Etape 11 - Lancer le pipeline

Lance toutes les etapes dans l'ordre : nettoyage, EDA, creation des DataLoaders.

```bash
python main.py
```

Resultat attendu :

```
=======================================================
  PFA 2A - Detection & Segmentation Lesions Cutanees
=======================================================

Etape 1 - Nettoyage des datasets
[OK] PH2 nettoye - 200 paires copiees
[OK] 200 masques binarises
[OK] 200 paires verifiees

Etape 2 - Analyse exploratoire
[OK] class_distribution.png sauvegarde
[OK] representative_grid.png sauvegarde
[OK] ham_image_quality.png sauvegarde
[OK] ph2_image_quality.png sauvegarde

Etape 3 - Creation des DataLoaders
HAM10000 - Train: 7010 | Val: 1502 | Test: 1503
PH2      - Train: 140  | Val: 30   | Test: 30

[OK] Pipeline de donnees pret !
```

### Ordre d'execution complet - resume

```bash
# Activer l'environnement (a faire a chaque nouveau terminal)
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Configurer le fichier .env (une seule fois)
copy .env.example .env         # Windows
cp .env.example .env           # Mac/Linux
# Puis remplir .env avec tes credentials Kaggle

# Telecharger les datasets (une seule fois)
python data/download.py

# Nettoyer PH2 (une seule fois)
python data/clean.py

# Analyse exploratoire
python utils/eda.py

# Pipeline complet
python main.py
```

---

## Verification PyTorch

### Verifier que PyTorch est correctement installe

```bash
python -c "import torch; print('Version PyTorch :', torch.__version__)"
```

### Verifier la disponibilite du GPU

```bash
python -c "
import torch
print('GPU disponible :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Nom du GPU     :', torch.cuda.get_device_name(0))
    print('Memoire GPU    :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"
```

### Verifier qu'un tenseur peut etre cree

```bash
python -c "
import torch
x = torch.rand(3, 224, 224)
print('Tenseur cree - Shape :', x.shape, '| dtype :', x.dtype)
"
```

### Tester le DataLoader HAM10000

```bash
python -c "
from preprocessing.datasets import get_ham10000_loaders
train, val, test, weights = get_ham10000_loaders()
images, labels = next(iter(train))
print('Batch images shape :', images.shape)
print('Batch labels shape :', labels.shape)
print('Class weights      :', weights.tolist())
"
```

Resultat attendu :
```
Batch images shape : torch.Size([8, 3, 224, 224])
Batch labels shape : torch.Size([8])
Class weights      : [0.21, 1.34, 1.36, 2.90, 4.57, 10.57, 13.07]
```

### Tester le DataLoader PH2

```bash
python -c "
from preprocessing.datasets import get_ph2_loaders
train, val, test = get_ph2_loaders()
images, masks = next(iter(train))
print('Batch images shape  :', images.shape)
print('Batch masques shape :', masks.shape)
print('Valeurs masque      :', masks.unique().tolist())
"
```

Resultat attendu :
```
Batch images shape  : torch.Size([8, 3, 224, 224])
Batch masques shape : torch.Size([8, 224, 224])
Valeurs masque      : [0, 1]
```

### Verifier toutes les versions installees

```bash
python -c "
import torch, torchvision, sklearn, pandas, matplotlib, seaborn, PIL
print('torch       :', torch.__version__)
print('torchvision :', torchvision.__version__)
print('sklearn     :', sklearn.__version__)
print('pandas      :', pandas.__version__)
print('matplotlib  :', matplotlib.__version__)
print('seaborn     :', seaborn.__version__)
print('PIL         :', PIL.__version__)
"
```

---

## Parametres configurables

Tous les parametres du projet sont dans config/config.py.
Modifier uniquement ce fichier pour changer le comportement global.

```python
IMAGE_SIZE    = (224, 224)   # Taille des images apres redimensionnement
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # Normalisation standard ImageNet
IMAGENET_STD  = [0.229, 0.224, 0.225]

TEST_SIZE     = 0.3          # 30% pour validation + test
VAL_SIZE      = 0.5          # 50% du temp = 15% total
RANDOM_STATE  = 42           # Seed pour reproductibilite

BATCH_SIZE    = 8            # Reduire a 4 ou 2 si erreur memoire GPU
NUM_WORKERS   = 2            # Mettre 0 sur Windows si erreur DataLoader
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
```

---

## Problemes frequents

### EnvironmentError : Variables KAGGLE_USERNAME et KAGGLE_KEY manquantes

Le fichier .env est absent ou mal configure.

```bash
# Creer le fichier .env depuis le template
copy .env.example .env         # Windows
cp .env.example .env           # Mac/Linux
```

Puis ouvrir .env et remplir les deux variables avec tes credentials Kaggle.

### ImportError : cannot import name 'X' from config.config

Le fichier config/config.py est une ancienne version.
Recuperer la derniere version depuis GitHub :

```bash
git pull origin main
```

### kaggle: command not found

```bash
pip install kaggle
```

### unzip se bloque sur Windows

Utiliser Python pour extraire l'archive :

```bash
python -c "import zipfile; zipfile.ZipFile('NOM_FICHIER.zip').extractall('DOSSIER_DESTINATION')"
```

### ModuleNotFoundError: No module named 'torch'

Le venv n'est pas active ou les dependances ne sont pas installees :

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### CUDA out of memory

Reduire la taille des batchs dans config/config.py :

```python
BATCH_SIZE = 4    # ou 2 si le probleme persiste
```

### RuntimeError: DataLoader worker process crashed (Windows)

Mettre NUM_WORKERS a 0 dans config/config.py :

```python
NUM_WORKERS = 0
```

### Session Google Colab reinitalisee (donnees perdues)

Monter Google Drive pour conserver les donnees entre les sessions :

```python
from google.colab import drive
drive.mount('/content/drive')
```

Puis modifier DATA_DIR dans config/config.py pour pointer vers Drive.

---

## Informations sur les datasets

### HAM10000 - Classification

| Classe  | Nom complet                  | Images | Pourcentage |
|---------|------------------------------|--------|-------------|
| nv      | Naevi melanocytaires         | 6705   | 66.9%       |
| mel     | Melanome                     | 1113   | 11.1%       |
| bkl     | Keratose benigne             | 1099   | 11.0%       |
| bcc     | Carcinome basocellulaire     | 514    | 5.1%        |
| akiec   | Keratose actinique           | 327    | 3.3%        |
| vasc    | Lesion vasculaire            | 142    | 1.4%        |
| df      | Dermatofibrome               | 115    | 1.1%        |

Le dataset est fortement desequilibre (nv = 66.9%).
Des poids de classes sont calcules automatiquement dans datasets.py et doivent
etre injectes dans la fonction de perte lors de l'entrainement :

```python
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
```

### PH2 - Segmentation

- 200 images dermoscopiques avec masques annotes manuellement
- Dimensions : environ 765 x 575 pixels
- Couverture moyenne de la lesion : 32.2% des pixels
- Masques binaires : 0 = fond, 1 = lesion

---

## Notes importantes

- Ne jamais versionner le dossier data/ sur GitHub (trop lourd, presente dans .gitignore)
- Ne jamais versionner le fichier .env sur GitHub (credentials Kaggle, presente dans .gitignore)
- Ne jamais versionner kaggle.json (cle API privee, presente dans .gitignore)
- Toujours activer le venv avant de lancer une commande Python
- Toujours creer son propre .env depuis .env.example avant de lancer le projet
- Les graphiques sont sauvegardes dans outputs/
- Les poids des modeles seront sauvegardes dans checkpoints/
- Modifier uniquement config/config.py pour changer les parametres globaux
