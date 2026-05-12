# Early Stopping + Model Checkpoint
# Surveille val_loss et sauvegarde le meilleur modèle

import torch
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience  = patience
        self.delta     = delta
        self.counter   = 0
        self.best_loss = None
        self.stop      = False

    def __call__(self, val_loss: float, model: torch.nn.Module, path: str) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save(model, path)

        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            logger.warning(f"EarlyStopping : patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
                logger.info("EarlyStopping : arret de l'entrainement")
        else:
            self.best_loss = val_loss
            self._save(model, path)
            self.counter = 0

    def _save(self, model: torch.nn.Module, path: str) -> None:
        torch.save(model.state_dict(), path)
        logger.info(f"Meilleur modele sauvegarde : {path}")