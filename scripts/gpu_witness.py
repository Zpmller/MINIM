#!/usr/bin/env python3
"""Seeded CUDA forward/backward smoke test for PyTorch and PyG GATv2."""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_minim_scorer import GATv2Scorer, MinimScorer


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(32, 512, device=device)
    edge_index = torch.randint(0, 32, (2, 128), device=device)
    model = GATv2Scorer(hidden=64, heads=4).to(device)
    s, n = model(x, edge_index)
    loss = (s.square().mean() + n.square().mean())
    loss.backward()
    if not torch.isfinite(loss) or not all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    ):
        raise SystemExit("non-finite loss or gradients")
    mlp = MinimScorer(hidden=16).to(device)
    ms, mn = mlp(torch.zeros(32, dtype=torch.long, device=device),
                 torch.zeros(32, dtype=torch.long, device=device),
                 torch.randn(32, 128, device=device), torch.randn(32, 4, device=device))
    (ms.mean() + mn.mean()).backward()
    torch.cuda.synchronize()
    print(f"WITNESS {torch.__version__=} {torch.version.cuda=} device={torch.cuda.get_device_name()} "
          f"gat_shape={tuple(s.shape)} loss={loss.item():.6f} "
          f"peak_mb={torch.cuda.max_memory_allocated() / 1024**2:.1f}")


if __name__ == "__main__":
    main()
