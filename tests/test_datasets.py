# tests/test_datasets.py
# ============================================================
# Tests unitaires des DataLoaders HAM10000 et PH2
# Usage : pytest tests/test_datasets.py -v
# ============================================================

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader


# ============================================================
# Tests HAM10000
# ============================================================

class TestHAM10000Loaders:

    @pytest.fixture(scope="class")
    def loaders(self):
        from preprocessing.datasets import get_ham10000_loaders
        train, val, test, weights = get_ham10000_loaders()
        return train, val, test, weights

    def test_train_batch_image_shape(self, loaders):
        train, _, _, _ = loaders
        images, labels = next(iter(train))
        assert images.shape[1:] == torch.Size([3, 224, 224]), \
            f"Shape attendu (B,3,224,224), recu {images.shape}"

    def test_train_batch_label_shape(self, loaders):
        train, _, _, _ = loaders
        images, labels = next(iter(train))
        assert labels.ndim == 1, \
            f"Labels doivent etre 1D, recu {labels.ndim}D"

    def test_labels_in_valid_range(self, loaders):
        train, _, _, _ = loaders
        images, labels = next(iter(train))
        assert labels.min() >= 0, "Label negatif detecte"
        assert labels.max() <= 6, f"Label > 6 detecte : {labels.max()}"

    def test_image_values_normalized(self, loaders):
        train, _, _, _ = loaders
        images, _ = next(iter(train))
        assert images.min() >= -3.0, "Valeur min anormale apres normalisation"
        assert images.max() <=  3.0, "Valeur max anormale apres normalisation"

    def test_val_batch_shape(self, loaders):
        _, val, _, _ = loaders
        images, labels = next(iter(val))
        assert images.shape[1:] == torch.Size([3, 224, 224])

    def test_test_batch_shape(self, loaders):
        _, _, test, _ = loaders
        images, labels = next(iter(test))
        assert images.shape[1:] == torch.Size([3, 224, 224])

    def test_class_weights_shape(self, loaders):
        _, _, _, weights = loaders
        assert weights.shape == torch.Size([7]), \
            f"Shape poids attendu (7,), recu {weights.shape}"

    def test_class_weights_positive(self, loaders):
        _, _, _, weights = loaders
        assert (weights > 0).all(), "Poids negatif ou nul detecte"

    def test_class_weights_rare_classes_higher(self, loaders):
        _, _, _, weights = loaders
        # df (index 6) doit avoir un poids plus eleve que nv (index 0)
        assert weights[6] > weights[0], \
            "Le poids de df doit etre superieur a nv"

    def test_no_data_leakage_train_val(self, loaders):
        train, val, _, _ = loaders
        train_paths = set(train.dataset.img_paths)
        val_paths   = set(val.dataset.img_paths)
        assert len(train_paths & val_paths) == 0, \
            "Data leakage detecte entre train et val"

    def test_no_data_leakage_train_test(self, loaders):
        train, _, test, _ = loaders
        train_paths = set(train.dataset.img_paths)
        test_paths  = set(test.dataset.img_paths)
        assert len(train_paths & test_paths) == 0, \
            "Data leakage detecte entre train et test"

    def test_no_data_leakage_val_test(self, loaders):
        _, val, test, _ = loaders
        val_paths  = set(val.dataset.img_paths)
        test_paths = set(test.dataset.img_paths)
        assert len(val_paths & test_paths) == 0, \
            "Data leakage detecte entre val et test"

    def test_split_proportions(self, loaders):
        train, val, test, _ = loaders
        total = len(train.dataset) + len(val.dataset) + len(test.dataset)
        train_ratio = len(train.dataset) / total
        val_ratio   = len(val.dataset)   / total
        test_ratio  = len(test.dataset)  / total
        assert abs(train_ratio - 0.70) < 0.02, \
            f"Train ratio attendu ~0.70, recu {train_ratio:.2f}"
        assert abs(val_ratio   - 0.15) < 0.02, \
            f"Val ratio attendu ~0.15, recu {val_ratio:.2f}"
        assert abs(test_ratio  - 0.15) < 0.02, \
            f"Test ratio attendu ~0.15, recu {test_ratio:.2f}"

    def test_image_dtype(self, loaders):
        train, _, _, _ = loaders
        images, _ = next(iter(train))
        assert images.dtype == torch.float32, \
            f"dtype attendu float32, recu {images.dtype}"


# ============================================================
# Tests PH2
# ============================================================

class TestPH2Loaders:

    @pytest.fixture(scope="class")
    def loaders(self):
        from preprocessing.datasets import get_ph2_loaders
        train, val, test = get_ph2_loaders()
        return train, val, test

    def test_train_image_shape(self, loaders):
        train, _, _ = loaders
        images, masks = next(iter(train))
        assert images.shape[1:] == torch.Size([3, 224, 224]), \
            f"Shape image attendu (B,3,224,224), recu {images.shape}"

    def test_train_mask_shape(self, loaders):
        train, _, _ = loaders
        images, masks = next(iter(train))
        assert masks.shape[1:] == torch.Size([224, 224]), \
            f"Shape masque attendu (B,224,224), recu {masks.shape}"

    def test_mask_binary_values(self, loaders):
        train, _, _ = loaders
        images, masks = next(iter(train))
        unique_values = masks.unique().tolist()
        assert all(v in [0, 1] for v in unique_values), \
            f"Masque non binaire detecte : valeurs {unique_values}"

    def test_mask_dtype(self, loaders):
        train, _, _ = loaders
        _, masks = next(iter(train))
        assert masks.dtype == torch.int64, \
            f"dtype masque attendu int64, recu {masks.dtype}"

    def test_image_dtype(self, loaders):
        train, _, _ = loaders
        images, _ = next(iter(train))
        assert images.dtype == torch.float32, \
            f"dtype image attendu float32, recu {images.dtype}"

    def test_val_shapes(self, loaders):
        _, val, _ = loaders
        images, masks = next(iter(val))
        assert images.shape[1:] == torch.Size([3, 224, 224])
        assert masks.shape[1:]  == torch.Size([224, 224])

    def test_test_shapes(self, loaders):
        _, _, test = loaders
        images, masks = next(iter(test))
        assert images.shape[1:] == torch.Size([3, 224, 224])
        assert masks.shape[1:]  == torch.Size([224, 224])

    def test_no_data_leakage_train_val(self, loaders):
        train, val, _ = loaders
        train_paths = set(train.dataset.img_paths)
        val_paths   = set(val.dataset.img_paths)
        assert len(train_paths & val_paths) == 0, \
            "Data leakage detecte entre train et val PH2"

    def test_no_data_leakage_train_test(self, loaders):
        train, _, test = loaders
        train_paths = set(train.dataset.img_paths)
        test_paths  = set(test.dataset.img_paths)
        assert len(train_paths & test_paths) == 0, \
            "Data leakage detecte entre train et test PH2"

    def test_total_size(self, loaders):
        train, val, test = loaders
        total = len(train.dataset) + len(val.dataset) + len(test.dataset)
        assert total == 200, \
            f"Total PH2 attendu 200, recu {total}"

    def test_image_mask_alignment(self, loaders):
        train, _, _ = loaders
        for i in range(min(5, len(train.dataset))):
            img_name  = os.path.basename(train.dataset.img_paths[i])
            mask_name = os.path.basename(train.dataset.mask_paths[i])
            assert img_name == mask_name, \
                f"Desalignement : image {img_name} vs masque {mask_name}"

    def test_mask_contains_lesion(self, loaders):
        train, _, _ = loaders
        images, masks = next(iter(train))
        # Au moins certains masques doivent contenir des pixels lésion
        assert masks.sum() > 0, \
            "Tous les masques sont vides (que des zeros)"


# ============================================================
# Tests modeles
# ============================================================

class TestModels:

    def test_classifier_output_shape(self):
        from models.classifier import build_classifier
        model  = build_classifier(num_classes=7)
        model.eval()
        x      = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == torch.Size([2, 7]), \
            f"Shape sortie classifieur attendu (2,7), recu {output.shape}"

    def test_segmentor_output_shape(self):
        from models.segmentor import build_segmentor
        model  = build_segmentor(num_classes=1)
        model.eval()
        x      = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == torch.Size([2, 1, 224, 224]), \
            f"Shape sortie segmenteur attendu (2,1,224,224), recu {output.shape}"

    def test_classifier_softmax_sums_to_one(self):
        from models.classifier import build_classifier
        import torch.nn.functional as F
        model  = build_classifier(num_classes=7)
        model.eval()
        x      = torch.rand(4, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
            probs  = F.softmax(output, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5), \
            "Softmax ne somme pas a 1"

    def test_segmentor_output_in_valid_range(self):
        from models.segmentor import build_segmentor
        model  = build_segmentor(num_classes=1)
        model.eval()
        x      = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            output = torch.sigmoid(model(x))
        assert output.min() >= 0.0, "Valeur sigmoid < 0"
        assert output.max() <= 1.0, "Valeur sigmoid > 1"


# ============================================================
# Tests losses
# ============================================================

class TestLosses:

    def test_focal_loss_perfect_prediction(self):
        from models.losses import FocalLoss
        criterion = FocalLoss(gamma=2.0)
        # Prediction parfaite
        inputs  = torch.tensor([[10.0, -10.0, -10.0]])
        targets = torch.tensor([0])
        loss    = criterion(inputs, targets)
        assert loss.item() < 0.01, \
            f"FocalLoss sur prediction parfaite trop haute : {loss.item()}"

    def test_focal_loss_wrong_prediction(self):
        from models.losses import FocalLoss
        criterion = FocalLoss(gamma=2.0)
        # Prediction inverse
        inputs  = torch.tensor([[-10.0, 10.0, -10.0]])
        targets = torch.tensor([0])
        loss    = criterion(inputs, targets)
        assert loss.item() > 1.0, \
            f"FocalLoss sur mauvaise prediction trop basse : {loss.item()}"

    def test_dice_bce_loss_perfect_mask(self):
        from models.losses import DiceBCELoss
        criterion = DiceBCELoss()
        pred   = torch.ones(2, 1, 224, 224) * 10.0
        target = torch.ones(2, 1, 224, 224)
        loss   = criterion(pred, target)
        assert loss.item() < 0.1, \
            f"DiceBCELoss sur masque parfait trop haute : {loss.item()}"

    def test_dice_bce_loss_empty_prediction(self):
        from models.losses import DiceBCELoss
        criterion = DiceBCELoss()
        pred   = torch.ones(2, 1, 224, 224) * -10.0   # predit tout a 0
        target = torch.ones(2, 1, 224, 224)
        loss   = criterion(pred, target)
        assert loss.item() > 1.0, \
            f"DiceBCELoss sur prediction vide trop basse : {loss.item()}"