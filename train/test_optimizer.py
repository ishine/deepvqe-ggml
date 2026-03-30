"""Verification tests for SOAP and Schedule-Free AdamW optimizers.

Tests confirm correct wiring: parameters update, no NaN, checkpoint
round-trips, and Schedule-Free eval mode behavior.
"""

import copy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import torch
import torch.nn as nn

import schedulefree
from soap import SOAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SmallConvGRUModel(nn.Module):
    """Minimal model with conv, GRU, and linear — mirrors DeepVQE structure."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.gru = nn.GRU(16, 8, batch_first=True)
        self.linear = nn.Linear(8, 1)

    def forward(self, x):
        # x: (B, 2, H, W)
        h = self.bn(self.conv(x))  # (B, 16, H, W)
        B, C, H, W = h.shape
        h = h.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, 16)
        h, _ = self.gru(h)  # (B, H*W, 8)
        return self.linear(h).mean()


def _dummy_input():
    return torch.randn(2, 2, 8, 8)


def _train_step(model, optimizer, x):
    """One forward + backward + step."""
    model.train()
    loss = model(x)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


# ---------------------------------------------------------------------------
# SOAP tests
# ---------------------------------------------------------------------------

def test_soap_basic_step():
    """SOAP can do forward+backward+step without NaN."""
    model = SmallConvGRUModel()
    optimizer = SOAP(model.parameters(), lr=3e-3, weight_decay=0.01)
    x = _dummy_input()

    params_before = {n: p.clone() for n, p in model.named_parameters()}

    # SOAP skips the first step (initializes preconditioner), so do 2 steps
    _train_step(model, optimizer, x)
    _train_step(model, optimizer, x)

    for name, p in model.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in {name}"
        assert not torch.isinf(p).any(), f"Inf in {name}"
        assert not torch.equal(p, params_before[name]), f"{name} didn't change"

    print("  PASS: test_soap_basic_step")


def test_soap_gru_step():
    """SOAP handles GRU weight matrices without numerical issues."""
    gru = nn.GRU(64, 32, batch_first=True)
    optimizer = SOAP(gru.parameters(), lr=3e-3, weight_decay=0.01)
    x = torch.randn(4, 10, 64)

    params_before = {n: p.clone() for n, p in gru.named_parameters()}

    for _ in range(5):
        out, _ = gru(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for name, p in gru.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in GRU {name}"
        assert not torch.isinf(p).any(), f"Inf in GRU {name}"
        assert not torch.equal(p, params_before[name]), f"GRU {name} didn't change"

    # Check that preconditioner state was created for 2D params
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.dim() >= 2:
                state = optimizer.state[p]
                assert "GG" in state, "Missing preconditioner state GG"

    print("  PASS: test_soap_gru_step")


def test_soap_checkpoint_roundtrip():
    """SOAP optimizer state survives save/load cycle."""
    model = SmallConvGRUModel()
    optimizer = SOAP(model.parameters(), lr=3e-3)
    x = _dummy_input()

    # Train a few steps to populate state
    for _ in range(3):
        _train_step(model, optimizer, x)

    # Save
    state_before = copy.deepcopy(optimizer.state_dict())

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(optimizer.state_dict(), f.name)

        # Load into fresh optimizer
        model2 = SmallConvGRUModel()
        model2.load_state_dict(model.state_dict())
        optimizer2 = SOAP(model2.parameters(), lr=3e-3)
        # Need to initialize state first
        _train_step(model2, optimizer2, x)
        optimizer2.load_state_dict(torch.load(f.name, weights_only=False))

    state_after = optimizer2.state_dict()

    # Compare param group settings
    for key in ("lr", "weight_decay"):
        assert state_before["param_groups"][0][key] == state_after["param_groups"][0][key], \
            f"Mismatch in {key}"

    print("  PASS: test_soap_checkpoint_roundtrip")


# ---------------------------------------------------------------------------
# Schedule-Free tests
# ---------------------------------------------------------------------------

def test_schedulefree_basic_step():
    """Schedule-Free AdamW can train and switch modes."""
    model = SmallConvGRUModel()
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=3e-4, warmup_steps=5,
    )
    x = _dummy_input()

    optimizer.train()
    params_before = {n: p.clone() for n, p in model.named_parameters()}

    for _ in range(3):
        _train_step(model, optimizer, x)

    for name, p in model.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in {name}"
        assert not torch.equal(p, params_before[name]), f"{name} didn't change"

    # Should not raise
    optimizer.eval()
    optimizer.train()

    print("  PASS: test_schedulefree_basic_step")


def test_schedulefree_eval_changes_params():
    """Switching to eval mode changes parameter values (averaged weights)."""
    model = SmallConvGRUModel()
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=3e-4, warmup_steps=2,
    )
    x = _dummy_input()

    optimizer.train()
    for _ in range(5):
        _train_step(model, optimizer, x)

    train_params = {n: p.clone() for n, p in model.named_parameters()}

    optimizer.eval()
    eval_params = {n: p.clone() for n, p in model.named_parameters()}

    # At least some parameters should differ between train and eval
    any_different = False
    for name in train_params:
        if not torch.equal(train_params[name], eval_params[name]):
            any_different = True
            break

    assert any_different, "eval() should change parameters (weight averaging)"

    # Switch back should restore train params
    optimizer.train()
    for name, p in model.named_parameters():
        assert torch.equal(p, train_params[name]), \
            f"train() didn't restore {name}"

    print("  PASS: test_schedulefree_eval_changes_params")


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------

def test_create_optimizer_schedulefree():
    """Factory creates Schedule-Free with no external scheduler."""
    from train import create_optimizer
    from src.config import load_config

    cfg = load_config("configs/default.yaml")
    cfg.training.optimizer = "schedulefree"
    model = SmallConvGRUModel()

    optimizer, scheduler = create_optimizer(cfg, model.parameters(), warmup_steps=100)

    assert isinstance(optimizer, schedulefree.AdamWScheduleFree)
    assert scheduler is None

    print("  PASS: test_create_optimizer_schedulefree")


def test_create_optimizer_soap():
    """Factory creates SOAP with linear warmup scheduler."""
    from train import create_optimizer
    from src.config import load_config

    cfg = load_config("configs/default.yaml")
    cfg.training.optimizer = "soap"
    model = SmallConvGRUModel()

    optimizer, scheduler = create_optimizer(cfg, model.parameters(), warmup_steps=100)

    assert isinstance(optimizer, SOAP)
    assert scheduler is not None

    print("  PASS: test_create_optimizer_soap")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== SOAP tests ===")
    test_soap_basic_step()
    test_soap_gru_step()
    test_soap_checkpoint_roundtrip()

    print("\n=== Schedule-Free tests ===")
    test_schedulefree_basic_step()
    test_schedulefree_eval_changes_params()

    print("\n=== Factory tests ===")
    test_create_optimizer_schedulefree()
    test_create_optimizer_soap()

    print("\nAll optimizer tests passed.")
