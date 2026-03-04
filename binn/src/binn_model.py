"""
BINN Model Architecture.

Biologically Informed Neural Network with sparse masked layers constrained
by Reactome pathway + STRING PPI connectivity matrices.

Activation: torch.nn.Mish  (smooth, self-gated; avoids dead neurons)
Optimizer:  torch.optim.Muon for 2D hidden weights  (Newton-Schulz orthogonalised
            momentum; added in PyTorch 2.9)
            torch.optim.AdamW for biases, BatchNorm, and output layer

IMPORTANT: Muon is the OPTIMIZER; Mish is the ACTIVATION.  Do not confuse them.

Layer order inside each SparseMaskedLayer:
    masked linear → Mish → BatchNorm1d → Dropout
"""
from __future__ import annotations

import logging
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MUON_AVAILABLE = hasattr(torch.optim, "Muon")
if not MUON_AVAILABLE:
    log.warning(
        f"torch.optim.Muon not available (PyTorch {torch.__version__}). "
        "PyTorch >= 2.9 is required. BINNTrainer will fall back to AdamW."
    )


# ── Sparse Masked Layer ───────────────────────────────────────────────────────

class SparseMaskedLayer(nn.Module):
    """
    A biologically constrained linear layer.

    Connectivity is enforced by a binary mask ``C^(ℓ) ∈ {0,1}^{n_in × n_out}``.
    Only positions where the mask is 1 are active; the rest are permanently zero.

    Autograd Note
    -------------
    In the forward pass we compute ``effective_weight = W * mask`` (Hadamard
    product).  Because ``∂(W·C)/∂W = C`` element-wise, the gradient for
    masked positions is automatically 0 — no explicit gradient hook is needed.
    However, some optimizers (e.g. Muon with Newton-Schulz) may spread gradient
    signal into masked positions via their internal momentum.  Call
    ``BINN.enforce_masks()`` after each optimizer step to re-zero any leakage.

    Forward pipeline (hidden layers)
    ----------------------------------
    x  →  masked linear  →  Mish  →  BatchNorm1d  →  Dropout  →  output

    Forward pipeline (output layer, ``is_output=True``)
    ---------------------------------------------------
    x  →  masked linear  →  output (raw logits)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        connectivity_mask: torch.Tensor,
        bias: bool = True,
        dropout_rate: float = config.DROPOUT_RATE,
        is_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_output = is_output

        # Learnable weight; shape (out_features, in_features) — PyTorch convention.
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Mask stored transposed so it matches the weight matrix shape.
        # connectivity_mask: (in_features, out_features)  →  mask: (out_features, in_features)
        # Store on CPU so _init_weights (called in __init__) works regardless of
        # where connectivity_mask lives.  model.to(device) moves all buffers later.
        self.register_buffer("mask", connectivity_mask.T.cpu().contiguous())


        # Non-output layers get activation stack
        if not is_output:
            self.activation = nn.Mish()
            self.bn = nn.BatchNorm1d(out_features)
            self.dropout = nn.Dropout(p=dropout_rate)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform init, then zero out any weight at a masked position."""
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.linear.weight.data *= self.mask

        if self.linear.bias is not None:
            # Standard fan_in bias init (same as nn.Linear default)
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        effective_weight = self.linear.weight * self.mask
        h = F.linear(x, effective_weight, self.linear.bias)
        if not self.is_output:
            h = self.activation(h)
            h = self.bn(h)
            h = self.dropout(h)
        return h

    def get_effective_weight(self) -> torch.Tensor:
        """Return the masked weight matrix (detached)."""
        return (self.linear.weight * self.mask).detach()

    def extra_repr(self) -> str:
        nnz = int(self.mask.sum().item())
        total = self.mask.numel()
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"connections={nnz}/{total} ({100*nnz/total:.1f}%), "
            f"output={self.is_output}"
        )


# ── Copy / Pass-Through Layer ─────────────────────────────────────────────────

class CopyLayer(nn.Module):
    """
    Identity pass-through layer for copy/padding nodes.

    Implements the same masked-linear interface as SparseMaskedLayer but:
    - No activation, BatchNorm, or Dropout.
    - Weights initialised to 1.0 at connected positions (near-identity).
    - Intended for layers where nodes merely relay their input to the next layer.

    In the BINN architecture, copy nodes are typically mixed with pathway nodes
    within the same layer, so the BINN model uses SparseMaskedLayer for all
    layers.  CopyLayer is provided for explicit use when an entire layer is
    composed of copy nodes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        connectivity_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.register_buffer("mask", connectivity_mask.T.cpu().contiguous())

        # Initialise to identity at connected positions
        with torch.no_grad():
            self.linear.weight.data.zero_()
            self.linear.weight.data += self.mask.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.linear.weight * self.mask)


# ── BINN ──────────────────────────────────────────────────────────────────────

class BINN(nn.Module):
    """
    Biologically Informed Neural Network.

    Builds one SparseMaskedLayer per connectivity matrix. All but the last
    matrix produce hidden layers (Mish → BN → Dropout). The last matrix
    drives the output layer (raw logit, no activation).

    Parameters
    ----------
    connectivity_matrices : list of torch.Tensor
        Binary masks, one per layer transition.  Each has shape (n_l, n_{l+1}).
        Must already be on config.DEVICE.
    layer_sizes : list of int
        Number of nodes per layer (length = len(connectivity_matrices) + 1).
    dropout_rate : float
        Dropout probability for hidden SparseMaskedLayers.
    """

    def __init__(
        self,
        connectivity_matrices: list[torch.Tensor],
        layer_sizes: list[int],
        dropout_rate: float = config.DROPOUT_RATE,
    ) -> None:
        super().__init__()
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate

        if len(connectivity_matrices) < 1:
            raise ValueError("Need at least one connectivity matrix.")

        # Hidden layers: all matrices except the last
        hidden = []
        for C in connectivity_matrices[:-1]:
            n_in, n_out = C.shape
            hidden.append(SparseMaskedLayer(n_in, n_out, C, dropout_rate=dropout_rate))
        self.hidden_layers = nn.ModuleList(hidden)

        # Output layer: last matrix, raw logits only
        C_out = connectivity_matrices[-1]
        n_in, n_out = C_out.shape
        self.output_layer = SparseMaskedLayer(
            n_in, n_out, C_out, dropout_rate=dropout_rate, is_output=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch_size, n_genes)

        Returns
        -------
        (batch_size, 1) — raw logits for HPV-positive class
        """
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    # ── Parameter groups for dual-optimizer setup ─────────────────────────────

    def get_2d_weight_params(self) -> list[nn.Parameter]:
        """2D hidden weight matrices — intended for Muon optimizer."""
        return [layer.linear.weight for layer in self.hidden_layers]

    def get_non_2d_params(self) -> list[nn.Parameter]:
        """All other parameters — intended for AdamW optimizer."""
        muon_ids = {id(p) for p in self.get_2d_weight_params()}
        return [p for p in self.parameters() if id(p) not in muon_ids]

    # ── Mask enforcement (use after each optimizer step) ──────────────────────

    def enforce_masks(self) -> None:
        """
        Re-zero masked weight positions after an optimizer step.

        Muon's Newton-Schulz orthogonalisation can spread momentum into masked
        positions.  This call ensures the sparse constraint is always respected
        in the network weights, not just in the forward pass.
        """
        with torch.no_grad():
            for layer in list(self.hidden_layers) + [self.output_layer]:
                layer.linear.weight.data *= layer.mask

    # ── Interpretability helpers ──────────────────────────────────────────────

    def get_layer_weights(self) -> list[torch.Tensor]:
        """
        Return the effective (masked) weight for every layer, detached.
        Index 0 = first hidden layer, last = output layer.
        """
        weights = [layer.get_effective_weight() for layer in self.hidden_layers]
        weights.append(self.output_layer.get_effective_weight())
        return weights


# ── Model Summary ─────────────────────────────────────────────────────────────

def print_model_summary(
    model: BINN,
    connectivity_matrices: list[torch.Tensor],
) -> None:
    """Print a human-readable architecture summary."""
    print()
    print("BINN Architecture Summary")
    print("===========================")

    layer_names = (
        ["Input (Genes)"]
        + [f"Hidden L{i+1}" for i in range(len(model.hidden_layers))]
        + ["Output"]
    )

    for i, (name, size) in enumerate(zip(layer_names, model.layer_sizes)):
        if i == 0:
            print(f"  Layer {i:2d} ({name:25s}):  {size:5d} nodes")
        else:
            C = connectivity_matrices[i - 1]
            nnz = int(C.sum().item())
            total = C.numel()
            sparsity = 100.0 * (1 - nnz / total)
            label = "Output" if i == len(model.layer_sizes) - 1 else name
            print(
                f"  Layer {i:2d} ({label:25s}):  {size:5d} nodes, "
                f"{nnz:6d} connections ({sparsity:.1f}% sparse)"
            )

    all_params = sum(p.numel() for p in model.parameters())
    params_2d = sum(p.numel() for p in model.get_2d_weight_params())
    params_other = sum(p.numel() for p in model.get_non_2d_params())
    total_mask_zeros = sum(
        int((layer.mask == 0).sum().item())
        for layer in list(model.hidden_layers) + [model.output_layer]
    )
    active_weights = all_params - total_mask_zeros
    dense_equiv = sum(
        a * b for a, b in zip(model.layer_sizes[:-1], model.layer_sizes[1:])
    )

    print()
    print(f"  Total trainable parameters:    {all_params:,}")
    print(f"    → 2D weights (Muon):         {params_2d:,}")
    print(f"    → Other params (AdamW):       {params_other:,}")
    print(f"  Masked (zero) weight positions: {total_mask_zeros:,}")
    print(f"  Active connections:             {active_weights:,}")
    print(
        f"  Effective vs dense equivalent:  {active_weights:,} / {dense_equiv:,} "
        f"({100*active_weights/dense_equiv:.1f}%)"
    )
    print(f"  Activation:                    Mish")
    optimizer_str = (
        "Muon (hidden 2D) + AdamW (bias/norm/output)"
        if MUON_AVAILABLE
        else "AdamW (Muon unavailable)"
    )
    print(f"  Optimizer:                     {optimizer_str}")
    print(f"  Device:                        {config.DEVICE}")
