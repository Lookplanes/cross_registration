#!/usr/bin/env python3
"""End-to-end integration test — minimal resources (CPU, tiny models, 2 epochs).

Creates synthetic data on-the-fly and runs the full translate→register→evaluate
pipeline.  Verifies that every script can import, initialise, train a few steps,
and produce output files — without touching real data or GPUs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure src/ is on PYTHONPATH
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _write_image(path: str, size: tuple[int, int] = (64, 64)) -> None:
    Image.fromarray(np.random.randint(0, 256, size, dtype=np.uint8)).save(path)


def _make_translation_data(root: str, n: int = 4) -> None:
    """Create trainA/ and trainB/ with *n* paired synthetic images."""
    for sub in ("trainA", "trainB"):
        d = os.path.join(root, sub)
        os.makedirs(d)
        for i in range(n):
            _write_image(os.path.join(d, f"{i:04d}.png"), (64, 64))


def _make_registration_data(root: str, n: int = 4) -> None:
    """Create MultiModalityPairedDataset layout with synthetic pairs."""
    for task in ("ch0_to_ch1", "ch1_to_ch0"):
        task_dir = os.path.join(root, task)
        for sub in ("moving", "fixed", "gt_flow", "valid_mask"):
            os.makedirs(os.path.join(task_dir, sub))
        for i in range(n):
            _write_image(os.path.join(task_dir, "moving", f"{i:04d}.png"), (64, 64))
            _write_image(os.path.join(task_dir, "fixed", f"{i:04d}.png"), (64, 64))
            # Flow
            np.save(os.path.join(task_dir, "gt_flow", f"{i:04d}.npy"),
                    np.random.randn(2, 64, 64).astype(np.float32) * 0.1)
            # Mask
            Image.fromarray(np.ones((64, 64), dtype=np.uint8) * 255).save(
                os.path.join(task_dir, "valid_mask", f"{i:04d}.png"))


def _make_eval_data(root: str, n: int = 2) -> None:
    """Create PairedImageFolderDataset layout for evaluation."""
    for sub in ("moving", "fixed", "gt_flow", "valid_mask"):
        os.makedirs(os.path.join(root, sub))
    for i in range(n):
        _write_image(os.path.join(root, "moving", f"{i:04d}.png"), (64, 64))
        _write_image(os.path.join(root, "fixed", f"{i:04d}.png"), (64, 64))
        np.save(os.path.join(root, "gt_flow", f"{i:04d}.npy"),
                np.random.randn(2, 64, 64).astype(np.float32) * 0.1)
        Image.fromarray(np.ones((64, 64), dtype=np.uint8) * 255).save(
            os.path.join(root, "valid_mask", f"{i:04d}.png"))


# ---------------------------------------------------------------------------
# Step runners (invoke scripts via subprocess)
# ---------------------------------------------------------------------------


def _run(step: str, cmd: list[str], timeout: int = 300) -> bool:
    print(f"\n{'=' * 55}")
    print(f"  {step}")
    print(f"  {' '.join(cmd)}")
    print(f"{'=' * 55}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    # Print last few lines
    lines = result.stdout.strip().split("\n")[-8:]
    for line in lines:
        print(f"  {line}")
    if result.returncode != 0:
        print(f"\n  ✗ FAILED (rc={result.returncode}, {elapsed:.0f}s)")
        print(f"  STDERR:\n{result.stderr[-500:]}")
        return False
    print(f"  ✓ OK ({elapsed:.0f}s)")
    return True


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="crossreg_e2e_")
    out = os.path.join(tmp, "output")
    python = sys.executable

    print(f"Temp: {tmp}")
    print(f"Python: {python}")

    # ------------------------------------------------------------------
    # Step 1: Create data
    # ------------------------------------------------------------------
    trans_data = os.path.join(tmp, "trans_data")
    reg_data = os.path.join(tmp, "reg_data")
    eval_data = os.path.join(tmp, "eval_data")
    _make_translation_data(trans_data, n=4)
    _make_registration_data(reg_data, n=4)
    _make_eval_data(eval_data, n=2)
    print("✓ Synthetic data created")

    # ------------------------------------------------------------------
    # Step 2: Translation training (2 epochs, tiny model)
    # ------------------------------------------------------------------
    t_save = os.path.join(out, "translation")
    if not _run("1. Translation Training", [
        python, str(_SCRIPTS / "train_translation.py"),
        "--dataroot", trans_data,
        "--name", "e2e_test",
        "--save-dir", t_save,
        "--input-nc", "1", "--output-nc", "1",
        "--ngf", "8", "--ndf", "8", "--netF-nc", "16",
        "--n-epochs", "2", "--n-epochs-decay", "0",
        "--load-size", "72", "--crop-size", "64",
        "--batch-size", "2",
        "--print-freq", "2",
        "--save-epoch-freq", "2",
        "--device", "cpu",
    ]):
        return 1

    # ------------------------------------------------------------------
    # Step 3: Registration training (2 epochs, tiny model)
    # ------------------------------------------------------------------
    r_save = os.path.join(out, "registration")
    if not _run("2. Registration Training", [
        python, str(_SCRIPTS / "train_registration.py"),
        "--train-dir", reg_data,
        "--val-dir", reg_data,
        "--save-dir", r_save,
        "--batch-size", "2",
        "--epochs", "2",
        "--lr", "1e-4",
        "--img-size", "64", "64",
        "--print-freq", "2",
        "--val-interval", "1",
        "--device", "cpu",
    ]):
        return 1

    # ------------------------------------------------------------------
    # Step 4: Evaluation
    # ------------------------------------------------------------------
    e_save = os.path.join(out, "evaluation")
    tran_ckpt = os.path.join(t_save, "e2e_test", "latest_checkpoint.pth")
    reg_ckpt = os.path.join(r_save, "experiments", "model_best.pth")

    if not _run("3. Evaluation", [
        python, str(_SCRIPTS / "evaluate.py"),
        "--data-dir", eval_data,
        "--cut-ckpt", tran_ckpt,
        "--transmorph-ckpt", reg_ckpt,
        "--output-dir", e_save,
        "--img-size", "64", "64",
        "--input-nc", "1", "--output-nc", "1",
        "--device", "cpu",
    ]):
        return 1

    # ------------------------------------------------------------------
    # Step 5: Verify outputs
    # ------------------------------------------------------------------
    print(f"\n{'=' * 55}")
    print("  Verifying outputs ...")
    checks = [
        ("Translation ckpt", tran_ckpt),
        ("Registration ckpt", reg_ckpt),
        ("Metrics JSON", os.path.join(e_save, "metrics.json")),
    ]
    all_ok = True
    for label, path in checks:
        ok = os.path.isfile(path)
        status = "✓" if ok else "✗ MISSING"
        print(f"  {status}  {label}: {path}")
        if not ok:
            all_ok = False

    if all_ok and os.path.isfile(os.path.join(e_save, "metrics.json")):
        with open(os.path.join(e_save, "metrics.json")) as f:
            metrics = json.load(f)
        n = metrics.get("n_samples", 0)
        agg = metrics.get("aggregates", {})
        print(f"  Samples evaluated: {n}")
        for k, v in agg.items():
            print(f"    {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    if all_ok:
        print(f"\n{'=' * 55}")
        print("  ✓ End-to-end pipeline test PASSED")
        print(f"  Output: {out}")
        print(f"{'=' * 55}")
        return 0
    else:
        print(f"\n  ✗ Some outputs missing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
