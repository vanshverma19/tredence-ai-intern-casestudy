"""
Self-Pruning Neural Network on CIFAR-10
========================================
Tredence AI Engineering Internship — Case Study

Architecture:
  - Custom PrunableLinear layers with learnable sigmoid gates
  - Sparsity regularization: Total Loss = CE + λ * L1(gates)
  - Trained on CIFAR-10 for three values of λ

Usage:
  pip install torch torchvision matplotlib
  python prunable_net.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# 1. PRUNABLE LINEAR LAYER
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that multiplies each weight by a
    learnable gate in [0, 1].  Gates are produced by applying a sigmoid to
    a raw (unconstrained) parameter tensor called `gate_scores`.

    Forward pass:
        gates        = sigmoid(gate_scores)          # ∈ (0, 1)
        pruned_w     = weight * gates                # element-wise
        out          = input @ pruned_w.T + bias

    Gradients flow through both `weight` and `gate_scores` via autograd
    because all operations (sigmoid, *, matmul) are differentiable.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight & bias — same init as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores: one scalar per weight, initialised near 0.5 after sigmoid
        # (small positive values → sigmoid ≈ 0.5 → gates start "half open")
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform init for weight (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates        = torch.sigmoid(self.gate_scores)   # shape: (out, in)
        pruned_w     = self.weight * gates               # element-wise
        return F.linear(x, pruned_w, self.bias)

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of gate values for this layer (always positive after sigmoid)."""
        return torch.sigmoid(self.gate_scores).sum()

    def gate_values(self) -> torch.Tensor:
        """Return detached gate values for analysis."""
        return torch.sigmoid(self.gate_scores).detach().cpu()

    def sparsity_fraction(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below `threshold` (considered 'pruned')."""
        gates = self.gate_values()
        return (gates < threshold).float().mean().item()


# ─────────────────────────────────────────────
# 2. NETWORK DEFINITION
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    A simple feed-forward network for CIFAR-10 (32×32×3 → 10 classes)
    built entirely from PrunableLinear layers.

    Architecture:
        Flatten → 3072
        PrunableLinear(3072, 1024) → BN → ReLU → Dropout
        PrunableLinear(1024, 512)  → BN → ReLU → Dropout
        PrunableLinear(512, 256)   → BN → ReLU → Dropout
        PrunableLinear(256, 10)    → logits
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 256)
        self.fc4 = PrunableLinear(256, 10)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # flatten
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)                  # raw logits

    def prunable_layers(self):
        """Yield all PrunableLinear sub-modules."""
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def sparsity_loss(self) -> torch.Tensor:
        """Sum L1 gate norms across all prunable layers."""
        return sum(layer.sparsity_loss() for layer in self.prunable_layers())

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Global fraction of pruned weights (gate < threshold)."""
        fracs = [layer.sparsity_fraction(threshold) for layer in self.prunable_layers()]
        return np.mean(fracs)

    def all_gate_values(self) -> torch.Tensor:
        """Concatenate all gate values across layers for histogram plotting."""
        return torch.cat([l.gate_values().flatten() for l in self.prunable_layers()])


# ─────────────────────────────────────────────
# 3. DATA LOADING
# ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 256, num_workers: int = 2):
    """Download and return CIFAR-10 train / test DataLoaders."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True,
                                             download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(root="./data", train=False,
                                             download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lambda_sparse, device):
    model.train()
    total_ce = total_sp = total_loss = correct = seen = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(imgs)
        ce_loss      = F.cross_entropy(logits, labels)
        sp_loss      = model.sparsity_loss()
        loss         = ce_loss + lambda_sparse * sp_loss

        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_ce   += ce_loss.item()  * bs
        total_sp   += sp_loss.item()  * bs
        total_loss += loss.item()     * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        seen       += bs

    n = len(loader.dataset)
    return {
        "ce_loss"   : total_ce   / n,
        "sp_loss"   : total_sp   / n,
        "total_loss": total_loss / n,
        "accuracy"  : correct    / n * 100,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = seen = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        seen    += imgs.size(0)
    return correct / seen * 100


# ─────────────────────────────────────────────
# 5. FULL EXPERIMENT
# ─────────────────────────────────────────────

def run_experiment(lambda_sparse: float,
                   epochs: int,
                   train_loader,
                   test_loader,
                   device,
                   verbose: bool = True):
    """Train one model with a given λ and return results."""
    print(f"\n{'='*55}")
    print(f"  λ = {lambda_sparse}  |  epochs = {epochs}")
    print(f"{'='*55}")

    model = SelfPruningNet(dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer,
                                lambda_sparse, device)
        scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            sp = model.overall_sparsity()
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"CE={stats['ce_loss']:.3f}  "
                  f"SP={stats['sp_loss']:.1f}  "
                  f"Train Acc={stats['accuracy']:.1f}%  "
                  f"Sparsity={sp*100:.1f}%")
        history.append(stats)

    test_acc = evaluate(model, test_loader, device)
    sparsity = model.overall_sparsity() * 100
    gate_vals = model.all_gate_values().numpy()

    print(f"\n  ✓ Test Accuracy : {test_acc:.2f}%")
    print(f"  ✓ Sparsity Level: {sparsity:.2f}%  "
          f"(gates < 1e-2)\n")

    return {
        "lambda"    : lambda_sparse,
        "test_acc"  : test_acc,
        "sparsity"  : sparsity,
        "gate_vals" : gate_vals,
        "history"   : history,
        "model"     : model,
    }


# ─────────────────────────────────────────────
# 6. PLOTTING
# ─────────────────────────────────────────────

def plot_gate_distributions(results: list, best_idx: int = 1):
    """
    Plot gate-value histograms for all λ settings in a single figure.
    A successful result shows a large spike near 0 (pruned) and a
    second cluster of larger values (active weights).
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4),
                             sharey=False)
    colors = ["#E63946", "#457B9D", "#2A9D8F"]

    for ax, res, color in zip(axes, results, colors):
        gates = res["gate_vals"]
        ax.hist(gates, bins=80, color=color, alpha=0.85, edgecolor="white", lw=0.3)
        ax.set_title(
            f"λ = {res['lambda']}\n"
            f"Acc = {res['test_acc']:.1f}%  |  Sparsity = {res['sparsity']:.1f}%",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Gate Value  σ(gate_score)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.axvline(0.01, color="black", ls="--", lw=1, label="prune threshold (1e-2)")
        ax.legend(fontsize=8)

    # Highlight the "best" model panel
    axes[best_idx].patch.set_facecolor("#FFFDE7")

    fig.suptitle("Distribution of Gate Values After Training\n"
                 "(spike near 0 = successful pruning)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("gate_distributions.png", dpi=150, bbox_inches="tight")
    print("  → Saved  gate_distributions.png")
    plt.show()


def plot_training_curves(results: list):
    """Plot CE loss over epochs for each λ."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#E63946", "#457B9D", "#2A9D8F"]
    for res, color in zip(results, colors):
        epochs = range(1, len(res["history"]) + 1)
        ce = [h["ce_loss"] for h in res["history"]]
        ax.plot(epochs, ce, color=color, lw=2, label=f"λ = {res['lambda']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss (train)")
    ax.set_title("Training Loss Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    print("  → Saved  training_curves.png")
    plt.show()


# ─────────────────────────────────────────────
# 7. RESULTS TABLE
# ─────────────────────────────────────────────

def print_results_table(results: list):
    print("\n" + "─" * 55)
    print(f"{'λ (lambda)':<15} {'Test Accuracy':>15} {'Sparsity Level (%)':>20}")
    print("─" * 55)
    for r in results:
        print(f"{r['lambda']:<15} {r['test_acc']:>14.2f}% {r['sparsity']:>19.2f}%")
    print("─" * 55)


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    # ── Hyperparameters ──────────────────────
    EPOCHS      = 30          # increase to 50–60 for better accuracy
    BATCH_SIZE  = 256
    LAMBDAS     = [1e-5, 1e-4, 5e-4]   # low / medium / high sparsity
    # ─────────────────────────────────────────

    device = (
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    results = []
    for lam in LAMBDAS:
        res = run_experiment(lam, EPOCHS, train_loader, test_loader, device)
        results.append(res)

    # Summary
    print_results_table(results)

    # Plots
    # Best model = highest test accuracy
    best_idx = max(range(len(results)), key=lambda i: results[i]["test_acc"])
    plot_gate_distributions(results, best_idx=best_idx)
    plot_training_curves(results)

    print("\nDone. Check gate_distributions.png and training_curves.png.")


if __name__ == "__main__":
    main()
