#!/usr/bin/env python3
"""One-command experiment runner: translate → register → evaluate.

Reads an experiment YAML config and executes the full pipeline.
Suitable for batch-running many modality pairs.

Usage::

    python scripts/run_experiment.py --config configs/experiments/2pm_to_confocal.yaml

Example YAML::

    name: 2pm_to_confocal
    source_modality: 2PM
    target_modality: Confocal
    data_root: /data2/wuyh
    train_dir: /data2/xujr/idr_data/Train_CrossModal_unsup_full
    val_dir: /data2/xujr/idr_data/Train_CrossModal/Val
    test_dir: /data2/xujr/idr_data/dataset_crossmodal_test/Test/ch1_to_ch0
    save_root: /data2/xujr/output_model/experiments
    img_size: [256, 256]
    input_nc: 1
    output_nc: 1

    translation:
      n_epochs: 200
      n_epochs_decay: 200
      ngf: 64
      ndf: 64
      lr: 0.0002
      lambda_GAN: 1.0
      lambda_NCE: 1.0

    registration:
      epochs: 400
      lr: 0.0001
      ncc_weight: 1.0
      reg_weight: 1.0
      batch_size: 64

    evaluate:
      device: cpu
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a full cross-modality experiment")
    p.add_argument("--config", required=True, help="Path to experiment YAML")
    p.add_argument("--skip-translation", action="store_true")
    p.add_argument("--skip-registration", action="store_true")
    p.add_argument("--skip-evaluation", action="store_true")
    return p.parse_args()


def _script(name: str) -> str:
    """Return absolute path to a script in this directory."""
    return str(Path(__file__).resolve().parent / name)


def _run(cmd: list[str], step: str) -> int:
    print(f"\n{'=' * 60}")
    print(f"  RUNNING: {step}")
    print(f"  {' '.join(cmd)}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ✗ {step} FAILED (code={result.returncode})")
    return result.returncode


def main() -> int:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    name = cfg["name"]
    save_root = cfg["save_root"]
    img_size = cfg.get("img_size", [256, 256])
    device = cfg.get("evaluate", {}).get("device", "cpu")

    # --- 1. Translation ---
    if not args.skip_translation and "translation" in cfg:
        t_cfg = cfg["translation"]
        tran_save = os.path.join(save_root, name, "translation")
        rc = _run([
            sys.executable, _script("train_translation.py"),
            "--dataroot", cfg.get("translation_dataroot", os.path.join(cfg["data_root"], "paired")),
            "--name", f"{name}_translation",
            "--save-dir", tran_save,
            "--input-nc", str(cfg.get("input_nc", 1)),
            "--output-nc", str(cfg.get("output_nc", 1)),
            "--ngf", str(t_cfg.get("ngf", 64)),
            "--ndf", str(t_cfg.get("ndf", 64)),
            "--n-epochs", str(t_cfg.get("n_epochs", 200)),
            "--n-epochs-decay", str(t_cfg.get("n_epochs_decay", 200)),
            "--lr", str(t_cfg.get("lr", 2e-4)),
            "--lambda-GAN", str(t_cfg.get("lambda_GAN", 1.0)),
            "--lambda-NCE", str(t_cfg.get("lambda_NCE", 1.0)),
            "--device", device,
        ], "1. Translation Training")
        if rc != 0:
            return rc

    # --- 2. Registration ---
    if not args.skip_registration and "registration" in cfg:
        r_cfg = cfg["registration"]
        reg_save = os.path.join(save_root, name, "registration")
        rc = _run([
            sys.executable, _script("train_registration.py"),
            "--train-dir", cfg["train_dir"],
            "--val-dir", cfg.get("val_dir", ""),
            "--save-dir", reg_save,
            "--batch-size", str(r_cfg.get("batch_size", 64)),
            "--epochs", str(r_cfg.get("epochs", 400)),
            "--lr", str(r_cfg.get("lr", 1e-4)),
            "--ncc-weight", str(r_cfg.get("ncc_weight", 1.0)),
            "--reg-weight", str(r_cfg.get("reg_weight", 1.0)),
            "--img-size", str(img_size[0]), str(img_size[1]),
            "--device", device,
        ], "2. Registration Training")
        if rc != 0:
            return rc

    # --- 3. Evaluation ---
    if not args.skip_evaluation and "test_dir" in cfg:
        eval_out = os.path.join(save_root, name, "evaluation")
        tran_ckpt = os.path.join(save_root, name, "translation", f"{name}_translation",
                                 "latest_checkpoint.pth")
        reg_ckpt = os.path.join(save_root, name, "registration",
                                "experiments", "model_best.pth")
        if not os.path.isfile(reg_ckpt):
            # Fallback: look for any .pth file
            exp_dir = os.path.join(save_root, name, "registration", "experiments")
            if os.path.isdir(exp_dir):
                pths = sorted([f for f in os.listdir(exp_dir) if f.endswith(".pth")])
                if pths:
                    reg_ckpt = os.path.join(exp_dir, pths[0])

        cmd = [
            sys.executable, _script("evaluate.py"),
            "--data-dir", cfg["test_dir"],
            "--cut-ckpt", tran_ckpt,
            "--transmorph-ckpt", reg_ckpt,
            "--output-dir", eval_out,
            "--img-size", str(img_size[0]), str(img_size[1]),
            "--input-nc", str(cfg.get("input_nc", 1)),
            "--output-nc", str(cfg.get("output_nc", 1)),
            "--device", device,
        ]
        if args.skip_translation:
            cmd.append("--no-pipeline")

        rc = _run(cmd, "3. Evaluation")
        if rc != 0:
            return rc

    print(f"\n{'=' * 60}")
    print(f"  Experiment '{name}' complete.")
    print(f"  Output: {os.path.join(save_root, name)}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
