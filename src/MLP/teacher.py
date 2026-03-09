import copy
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
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

    def save(self, path: str | Path):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
