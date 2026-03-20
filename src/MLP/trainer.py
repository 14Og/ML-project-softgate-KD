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

    def _kd_loss_per_sample(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence per sample at temperature T. Shape: (batch,)."""
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        # sum over classes, keep per-sample
        return F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct = 0.0, 0
        T = self.temperature

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            student_logits = self.model(X)

            with torch.no_grad():
                teacher_logits = self.teacher(X)

            ce_loss = F.cross_entropy(student_logits, y)
            kd_loss = self._kd_loss_per_sample(student_logits, teacher_logits).mean()

            loss = (1 - self.alpha) * ce_loss + self.alpha * (T ** 2) * kd_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (student_logits.argmax(dim=1) == y).sum().item()

        n = len(loader.dataset)
        return total_loss / n, correct / n


class HKDTrainer(KDTrainer):
    """Hard Gate Knowledge Distillation (Lee et al. 2022).

    Per-sample binary gate based on calibration discrepancy:
      Δ = p_student(predicted_class) - p_teacher(predicted_class)

    If Δ > 0  → student is overconfident → α=1 → use KD loss (regularize)
    If Δ ≤ 0  → student is underconfident → α=0 → use CE loss (learn)

    Each sample either gets full teacher supervision or full ground-truth
    supervision — never a mix.
    """

    def __init__(
        self,
        model: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        temperature: float = 4.0,
    ):
        # alpha is unused — the gate replaces it entirely
        super().__init__(model, teacher, optimizer, device, alpha=0.0, temperature=temperature)

    def _compute_gate(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Binary gate: 1 where student is overconfident, 0 otherwise. Shape: (batch,)."""
        student_probs = F.softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)

        # Compare confidences on the class the student predicts
        student_pred = student_logits.argmax(dim=1)
        p_student = student_probs.max(dim=1).values
        p_teacher = teacher_probs.gather(1, student_pred.unsqueeze(1)).squeeze(1)

        return (p_student > p_teacher).float()

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct = 0.0, 0
        total_gate_on = 0  # track how often KD is applied
        T = self.temperature

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            student_logits = self.model(X)

            with torch.no_grad():
                teacher_logits = self.teacher(X)

            gate = self._compute_gate(student_logits, teacher_logits)  # (batch,)
            total_gate_on += gate.sum().item()

            # Per-sample losses
            ce_loss = F.cross_entropy(student_logits, y, reduction="none")  # (batch,)
            kd_loss = self._kd_loss_per_sample(student_logits, teacher_logits)  # (batch,)

            # gate=1 → KD, gate=0 → CE
            loss = (gate * (T ** 2) * kd_loss + (1 - gate) * ce_loss).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (student_logits.argmax(dim=1) == y).sum().item()

        n = len(loader.dataset)
        return total_loss / n, correct / n


class SKDTrainer(KDTrainer):
    """Soft Gate Knowledge Distillation.

    Extends HKD by replacing the binary gate with a smooth function
    of the calibration discrepancy:
        Δ = p_student(predicted_class) - p_teacher(predicted_class)

    Gate functions:
      - "sigmoid":  α = σ(sharpness · Δ)
      - "linear":   α = clamp(0.5 + sharpness · Δ,  0, 1)

    When α → 1 (overconfident): more teacher supervision (regularize)
    When α → 0 (underconfident): more ground-truth supervision (learn)

    As sharpness → ∞, the sigmoid gate converges to the hard gate (HKD).
    """

    def __init__(
        self,
        model: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        temperature: float = 4.0,
        gate_fn: str = "sigmoid",
        sharpness: float = 10.0,
    ):
        super().__init__(model, teacher, optimizer, device, alpha=0.0, temperature=temperature)
        self.gate_fn = gate_fn
        self.sharpness = sharpness

    def _compute_gate(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Smooth gate in [0, 1] based on calibration discrepancy. Shape: (batch,)."""
        student_probs = F.softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)

        student_pred = student_logits.argmax(dim=1)
        p_student = student_probs.max(dim=1).values
        p_teacher = teacher_probs.gather(1, student_pred.unsqueeze(1)).squeeze(1)

        delta = p_student - p_teacher

        if self.gate_fn == "sigmoid":
            return torch.sigmoid(self.sharpness * delta)
        elif self.gate_fn == "linear":
            return torch.clamp(0.5 + self.sharpness * delta, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown gate function: {self.gate_fn}")

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct = 0.0, 0
        T = self.temperature

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            student_logits = self.model(X)

            with torch.no_grad():
                teacher_logits = self.teacher(X)

            alpha = self._compute_gate(student_logits, teacher_logits).detach()

            # Per-sample losses
            ce_loss = F.cross_entropy(student_logits, y, reduction="none")
            kd_loss = self._kd_loss_per_sample(student_logits, teacher_logits)

            # Smooth mix: each sample gets its own alpha
            loss = (alpha * (T ** 2) * kd_loss + (1 - alpha) * ce_loss).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (student_logits.argmax(dim=1) == y).sum().item()

        n = len(loader.dataset)
        return total_loss / n, correct / n
