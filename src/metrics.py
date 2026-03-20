"""
Metrics for evaluating classification quality and calibration.

All functions take:
  logits : torch.Tensor of shape (N, C)  — raw model outputs (before softmax)
  labels : torch.Tensor of shape (N,)    — integer ground-truth class indices
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class MetricBundle:
    accuracy: float
    nll:      float
    ece:      float

    def __repr__(self):
        return (
            f"Accuracy: {self.accuracy:.4f} | "
            f"NLL: {self.nll:.4f} | "
            f"ECE: {self.ece:.4f}"
        )


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Average negative log-likelihood (= cross-entropy on test set).
    Penalises confident wrong predictions more than accuracy does.
    """
    return F.cross_entropy(logits, labels).item()


def ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error.

    Splits predictions into confidence bins and computes the weighted average
    of |confidence - accuracy| across bins. Lower is better (0 = perfect).
    """
    probs      = F.softmax(logits, dim=1)
    confidence = probs.max(dim=1).values          # max probability = model confidence
    correct    = (logits.argmax(dim=1) == labels).float()

    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece_val   = 0.0
    n         = len(labels)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        bin_confidence = confidence[mask].mean().item()
        bin_accuracy   = correct[mask].mean().item()
        bin_weight     = mask.sum().item() / n
        ece_val += bin_weight * abs(bin_confidence - bin_accuracy)

    return ece_val


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> MetricBundle:
    return MetricBundle(
        accuracy=accuracy(logits, labels),
        nll=nll(logits, labels),
        ece=ece(logits, labels),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
    title: str = "Reliability Diagram",
):
    """Plots confidence (x) vs accuracy (y) per bin.
    A perfectly calibrated model follows the diagonal.
    Bars below diagonal = overconfident; above = underconfident.
    """
    probs      = F.softmax(logits, dim=1)
    confidence = probs.max(dim=1).values.numpy()
    correct    = (logits.argmax(dim=1) == labels).float().numpy()

    bin_edges      = np.linspace(0, 1, n_bins + 1)
    bin_centers    = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accuracies = np.zeros(n_bins)
    bin_counts     = np.zeros(n_bins)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() > 0:
            bin_accuracies[i] = correct[mask].mean()
            bin_counts[i]     = mask.sum()

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    width = 1.0 / n_bins
    ax.bar(bin_centers, bin_accuracies, width=width * 0.9, alpha=0.7, label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax


def plot_confidence_histogram(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ax: plt.Axes | None = None,
    title: str = "Confidence Distribution",
):
    """Histogram of max(softmax) for correct vs incorrect predictions.
    Overfit models pile up near 1.0 even on wrong predictions.
    """
    probs      = F.softmax(logits, dim=1)
    confidence = probs.max(dim=1).values.numpy()
    correct    = (logits.argmax(dim=1) == labels).numpy().astype(bool)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    bins = np.linspace(0, 1, 21)
    ax.hist(confidence[correct],  bins=bins, alpha=0.6, label="Correct",   density=True)
    ax.hist(confidence[~correct], bins=bins, alpha=0.6, label="Incorrect", density=True)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    return ax


def plot_training_curves(history, ax1: plt.Axes | None = None, ax2: plt.Axes | None = None, label: str = ""):
    """Plots loss and accuracy curves from a History object."""
    if ax1 is None or ax2 is None:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history.train_loss) + 1)
    ax1.plot(epochs, history.train_loss, label=f"{label} train")
    ax1.plot(epochs, history.val_loss,   label=f"{label} val", linestyle="--")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, history.train_acc, label=f"{label} train")
    ax2.plot(epochs, history.val_acc,   label=f"{label} val", linestyle="--")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

    return ax1, ax2


def compare_metrics(results: dict[str, MetricBundle]):
    """Print a comparison table for multiple model variants."""
    print(f"{'Model':<20} {'Accuracy':>10} {'NLL':>10} {'ECE':>10}")
    print("-" * 52)
    for name, m in results.items():
        print(f"{name:<20} {m.accuracy:>10.4f} {m.nll:>10.4f} {m.ece:>10.4f}")
