# conftest.py
# ============================================================
# Configuration pytest — ajoute la racine du projet au path
# Permet aux tests de trouver les modules preprocessing, models
# ============================================================

import sys
import os

# Ajouter la racine du projet au path Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))