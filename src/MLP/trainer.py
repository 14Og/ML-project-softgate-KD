import copy
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)

    def update(self, tr_loss, tr_acc, va_loss, va_acc):
        self.train_loss.append(tr_loss)
        self.train_acc.append(tr_acc)
        self.val_loss.append(va_loss)
        self.val_acc.append(va_acc)

    @property
    def best_val_acc(self) -> float:
        return max(self.val_acc) if self.val_acc else 0.0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = History()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 5,
        verbose: bool = True,
        log_every: int = 5,
    ) -> History:
        """Train with early stopping on val loss.

        Saves the best model weights internally and restores them after training.
        """
        best_val_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._train_epoch(train_loader)
            va_loss, va_acc = self.evaluate(val_loader)
            self.history.update(tr_loss, tr_acc, va_loss, va_acc)

            if verbose and epoch % log_every == 0:
                print(
                    f"Epoch {epoch:3d} | "
                    f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                    f"val loss {va_loss:.4f} acc {va_acc:.4f}"
                )

            # Early stopping: track best val loss and save best weights
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (patience={patience})")
                    break

        # Restore best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            if verbose:
                print(f"Restored best model (val loss {best_val_loss:.4f})")

        return self.history

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct = 0.0, 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()

        n = len(loader.dataset)
        return total_loss / n, correct / n

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct = 0.0, 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()

        n = len(loader.dataset)
        return total_loss / n, correct / n

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (all_logits, all_labels) for the full loader.
        Used to compute metrics after training.
        """
        self.model.eval()
        all_logits, all_labels = [], []

        for X, y in loader:
            X = X.to(self.device)
            all_logits.append(self.model(X).cpu())
            all_labels.append(y)

        return torch.cat(all_logits), torch.cat(all_labels)

    def save(self, path: str | Path):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


class KDTrainer(Trainer):
    """Standard Knowledge Distillation (Hinton et al. 2015).

    Loss = (1 - alpha) * CE(student, hard_labels)
         + alpha * T^2 * KL(student_soft || teacher_soft)

    The T^2 factor compensates for the 1/T^2 gradient scaling introduced
    by temperature — without it, the KD term would be effectively shrunk
    relative to the CE term as T increases.
    """

    def __init__(
        self,
        model: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        alpha: float = 0.5,
        temperature: float = 4.0,
    ):
        # CE loss for the hard-label term
        super().__init__(model, optimizer, nn.CrossEntropyLoss(), device)
        self.teacher = teacher.to(device)
        self.teacher.eval()  # teacher is always frozen
        self.alpha = alpha
        self.temperature = temperature

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct = 0.0, 0
        T = self.temperature

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            student_logits = self.model(X)

            with torch.no_grad():
                teacher_logits = self.teacher(X)

            # Hard-label term
            ce_loss = self.criterion(student_logits, y)

            # Soft-label term: KL(student || teacher) at temperature T
            student_log_probs = F.log_softmax(student_logits / T, dim=1)
            teacher_probs     = F.softmax(teacher_logits / T, dim=1)
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

            loss = (1 - self.alpha) * ce_loss + self.alpha * (T ** 2) * kd_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (student_logits.argmax(dim=1) == y).sum().item()

        n = len(loader.dataset)
        return total_loss / n, correct / n
