"""
Training utilities for the BINN.

EarlyStopping  — monitors validation loss, saves / restores the best checkpoint.
BINNTrainer    — wraps the dual-optimizer (Muon + AdamW) training loop,
                 learning-rate schedulers, class weighting, and inference.

Dual-optimizer convention
--------------------------
Muon  → 2D hidden weight matrices (SparseMaskedLayer.linear.weight)
AdamW → everything else: biases, BatchNorm γ/β, output layer weight + bias
"""
from __future__ import annotations

import copy
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.binn_model import BINN, MUON_AVAILABLE

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Set all random seeds for reproducibility (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS does not expose a dedicated seed function; torch.manual_seed covers it.


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Monitor validation loss and stop when it fails to improve.

    Usage
    -----
    >>> es = EarlyStopping(patience=20)
    >>> for epoch in ...:
    ...     es(val_loss, model)
    ...     if es.should_stop:
    ...         break
    >>> es.restore_best(model)     # loads state from the best epoch

    Parameters
    ----------
    patience  : int   — epochs without improvement before stopping
    min_delta : float — minimum decrease to count as improvement
    """

    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state: dict | None = None
        self.should_stop = False

    def __call__(self, val_loss: float, model: nn.Module, epoch: int = 0) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module) -> None:
        """Load the model state from the best epoch."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            log.info(f"Restored model from best epoch {self.best_epoch} (val_loss={self.best_loss:.4f})")


# ── Trainer ───────────────────────────────────────────────────────────────────

class BINNTrainer:
    """
    Manages the full training lifecycle for a BINN model.

    Optimizer Setup
    ---------------
    Uses Muon for 2D hidden weight matrices and AdamW for everything else.
    Falls back to AdamW-only if torch.optim.Muon is unavailable or raises
    on the current device.

    Parameters
    ----------
    model      : BINN instance (already on config.DEVICE)
    device     : torch.device
    lr_muon    : learning rate for Muon (recommend 0.02)
    lr_adam    : learning rate for AdamW (recommend 1e-3)
    weight_decay : weight decay for Muon (AdamW uses config.ADAM_WEIGHT_DECAY)
    epochs     : maximum training epochs
    patience   : early stopping patience
    batch_size : mini-batch size
    """

    def __init__(
        self,
        model: BINN,
        device: torch.device | None = None,
        lr_muon: float | None = None,
        lr_adam: float | None = None,
        weight_decay: float | None = None,
        epochs: int | None = None,
        patience: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        if device is None:
            device = config.DEVICE
        if lr_muon is None:
            lr_muon = config.MUON_LR
        if lr_adam is None:
            lr_adam = config.ADAM_LR
        if weight_decay is None:
            weight_decay = config.MUON_WEIGHT_DECAY
        if epochs is None:
            epochs = config.EPOCHS
        if patience is None:
            patience = config.EARLY_STOPPING_PATIENCE
        if batch_size is None:
            batch_size = config.BATCH_SIZE

        self.model = model.to(device)
        self.device = device
        self.lr_muon = lr_muon
        self.lr_adam = lr_adam
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size

        # Optimizers and schedulers initialised lazily in fit()
        self.optimizer_muon: torch.optim.Optimizer | None = None
        self.optimizer_adam: torch.optim.Optimizer | None = None
        self.scheduler_muon: ReduceLROnPlateau | None = None
        self.scheduler_adam: ReduceLROnPlateau | None = None
        self.optimizer_type: str = "unknown"

    # ── Optimizer setup ───────────────────────────────────────────────────────

    def _setup_optimizers(self) -> None:
        """
        Partition parameters into Muon (2D hidden weights) and AdamW (rest).

        Muon notes
        ----------
        - Requires parameter tensors that are 2-D and on a supported device.
        - On MPS, bfloat16 is available since PyTorch 2.1, so Newton-Schulz
          should work.  A try/except guards against unexpected failures.
        - If Muon is unavailable or fails, falls back to AdamW for all params.
        """
        hidden_2d = self.model.get_2d_weight_params()
        other = self.model.get_non_2d_params()

        # Guard: empty param lists are not accepted by optimizers.
        if not hidden_2d:
            log.warning("No 2D hidden weights found; using AdamW for all parameters.")
            # Use a dummy tensor for optimizer_muon so that the unconditional
            # optimizer_adam assignment below exclusively handles all real params.
            # Both optimizers share the same zero_grad / step API in the training
            # loop; the dummy optimizer has no effect on model weights.
            self.optimizer_muon = torch.optim.AdamW(
                [torch.zeros(1, requires_grad=True)],  # dummy to keep uniform API
                lr=self.lr_adam,
            )
            self.optimizer_type = "adamw_only"
        elif MUON_AVAILABLE and hidden_2d and hidden_2d[0].device.type in {"cuda", "mps"}:
            try:
                self.optimizer_muon = torch.optim.Muon(
                    hidden_2d,
                    lr=self.lr_muon,
                    momentum=config.MUON_MOMENTUM,
                    weight_decay=self.weight_decay,
                    nesterov=config.MUON_NESTEROV,
                )
                self.optimizer_type = "muon+adamw"
                log.info("Using Muon optimizer for 2D hidden weights.")
            except (RuntimeError, AttributeError, TypeError) as exc:
                log.warning(
                    f"Muon initialisation failed ({exc}). Falling back to AdamW."
                )
                self.optimizer_muon = torch.optim.AdamW(
                    hidden_2d, lr=self.lr_adam,
                    weight_decay=config.ADAM_WEIGHT_DECAY, betas=config.ADAM_BETAS,
                )
                self.optimizer_type = "adamw_fallback"
        elif MUON_AVAILABLE:
            log.info(
                "Using AdamW fallback for hidden 2D weights on CPU "
                "(Muon is only enabled on CUDA/MPS)."
            )
            self.optimizer_muon = torch.optim.AdamW(
                hidden_2d,
                lr=self.lr_adam,
                weight_decay=config.ADAM_WEIGHT_DECAY,
                betas=config.ADAM_BETAS,
            )
            self.optimizer_type = "adamw_fallback"
        else:
            log.warning(
                f"torch.optim.Muon unavailable (PyTorch {torch.__version__}). "
                "Using AdamW for all parameters."
            )
            self.optimizer_muon = torch.optim.AdamW(
                hidden_2d, lr=self.lr_adam,
                weight_decay=config.ADAM_WEIGHT_DECAY, betas=config.ADAM_BETAS,
            )
            self.optimizer_type = "adamw_fallback"

        self.optimizer_adam = torch.optim.AdamW(
            other,
            lr=self.lr_adam,
            weight_decay=config.ADAM_WEIGHT_DECAY,
            betas=config.ADAM_BETAS,
        )

        self.scheduler_muon = ReduceLROnPlateau(
            self.optimizer_muon, mode="min", factor=0.5, patience=10
        )
        self.scheduler_adam = ReduceLROnPlateau(
            self.optimizer_adam, mode="min", factor=0.5, patience=10
        )

    # ── Training loop ─────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """
        Train the BINN with early stopping and dual-optimizer schedule.

        Parameters
        ----------
        X_train, X_val : (n_samples, n_genes) float32 arrays
        y_train, y_val : (n_samples,) binary int arrays

        Returns
        -------
        dict with keys:
            train_loss, val_loss, val_auroc   — per-epoch lists
            best_epoch                         — epoch of best validation loss
            lr_muon_history, lr_adam_history  — per-epoch LR lists
            optimizer_type                     — "muon+adamw" or "adamw_fallback"
        """
        set_seeds(config.RANDOM_SEED)
        self._setup_optimizers()

        # ── Class-weighted loss ──────────────────────────────────────────
        pos_count = float(y_train.sum())
        neg_count = float(len(y_train) - pos_count)
        pos_weight = torch.tensor(
            [neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=self.device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # ── DataLoader ───────────────────────────────────────────────────
        X_t = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y_train.astype(np.float32)).to(self.device).unsqueeze(1)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        X_v = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
        y_v_t = torch.from_numpy(y_val.astype(np.float32)).to(self.device).unsqueeze(1)

        history: dict = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
            "best_epoch": 0,
            "lr_muon_history": [],
            "lr_adam_history": [],
            "optimizer_type": self.optimizer_type,
        }

        early_stopping = EarlyStopping(patience=self.patience)

        for epoch in range(self.epochs):
            # ── Train ────────────────────────────────────────────────────
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                self.optimizer_muon.zero_grad()
                self.optimizer_adam.zero_grad()

                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()

                self.optimizer_muon.step()
                self.optimizer_adam.step()

                # Re-zero any weights that Muon's Newton-Schulz may have
                # leaked into masked positions.
                self.model.enforce_masks()

                epoch_loss += loss.item() * len(X_batch)

            train_loss = epoch_loss / len(X_train)

            # ── Validate ─────────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_v)
                val_loss = criterion(val_logits, y_v_t).item()
                val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()

            val_auroc = (
                roc_auc_score(y_val, val_probs)
                if len(np.unique(y_val)) > 1
                else 0.5
            )

            # ── LR schedulers ─────────────────────────────────────────────
            self.scheduler_muon.step(val_loss)
            self.scheduler_adam.step(val_loss)

            lr_muon = self.optimizer_muon.param_groups[0]["lr"]
            lr_adam = self.optimizer_adam.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auroc"].append(val_auroc)
            history["lr_muon_history"].append(lr_muon)
            history["lr_adam_history"].append(lr_adam)

            if epoch % 10 == 0 or epoch < 5:
                log.info(
                    f"Epoch {epoch:3d}: train={train_loss:.4f}  "
                    f"val={val_loss:.4f}  AUROC={val_auroc:.4f}  "
                    f"lr_muon={lr_muon:.2e}  lr_adam={lr_adam:.2e}"
                )

            # ── Early stopping ────────────────────────────────────────────
            early_stopping(val_loss, self.model, epoch)
            if early_stopping.should_stop:
                log.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch: {early_stopping.best_epoch}."
                )
                break

        early_stopping.restore_best(self.model)
        history["best_epoch"] = early_stopping.best_epoch
        return history

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability of the positive class (HPV-positive) for each sample.

        Parameters
        ----------
        X : (n_samples, n_genes) float32 array

        Returns
        -------
        (n_samples,) float32 array in [0, 1]
        """
        self.model.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        return probs.astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Return binary class predictions.

        Parameters
        ----------
        X         : (n_samples, n_genes) float32 array
        threshold : classification threshold (default 0.5)

        Returns
        -------
        (n_samples,) int array of 0/1 predictions
        """
        return (self.predict_proba(X) >= threshold).astype(int)
