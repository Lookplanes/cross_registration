#!/usr/bin/env python3
"""Evaluate cross-modality registration pipeline on a test dataset.

Reports per-sample and aggregated metrics: ZNCC, MSE, NMI, Foreground Dice,
EPE, and folding ratio.

Usage::

    python scripts/evaluate.py \
        --data-dir /path/to/test/ch1_to_ch0 \
        --cut-ckpt /path/to/cut_latest.pth \
        --transmorph-ckpt /path/to/transmorph_best.pth \
        --output-dir results/eval_output \
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crossreg.data.datasets import PairedImageFolderDataset
from crossreg.registration.transmorph.model import TransMorph, CONFIGS
from crossreg.translation.cut import CUTInference
from crossreg.pipeline.inference import PipelineInference
from crossreg.utils.metrics import (
    AverageMeter,
    compute_zncc,
    compute_mse,
    compute_nmi,
    compute_foreground_dice,
    compute_epe,
    compute_folding_ratio,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate registration pipeline")
    p.add_argument("--data-dir", required=True,
                   help="Test dataset root (with moving/ fixed/ gt_flow/ valid_mask/)")
    p.add_argument("--cut-ckpt", required=True, help="Path to CUT Generator weights")
    p.add_argument("--transmorph-ckpt", required=True, help="Path to TransMorph weights")
    p.add_argument("--output-dir", default="results/eval")
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--input-nc", type=int, default=1)
    p.add_argument("--output-nc", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-pipeline", action="store_true",
                   help="Skip CUT translation, evaluate TransMorph only")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_cut(path: str, input_nc: int, output_nc: int, device: torch.device) -> CUTInference:
    model = CUTInference(
        input_nc=input_nc, output_nc=output_nc,
        gpu_ids=[0] if device.type == "cuda" else [],
    ).to(device)
    if os.path.isfile(path):
        state = torch.load(path, map_location=device, weights_only=True)
        model.netG.load_state_dict(state, strict=False)
        print(f"Loaded CUT from {path}")
    else:
        print(f"WARNING: CUT checkpoint not found at {path}, using random init")
    model.eval()
    return model


def load_transmorph(path: str, img_size: tuple[int, int],
                    in_chans: int, device: torch.device) -> TransMorph:
    config = CONFIGS["TransMorph"]
    config.img_size = img_size
    config.in_chans = in_chans
    model = TransMorph(config).to(device)

    if os.path.isfile(path):
        ck = torch.load(path, map_location=device, weights_only=False)
        state = ck.get("state_dict", ck)
        # Strip 'module.' prefix from DataParallel
        from collections import OrderedDict
        clean = OrderedDict()
        for k, v in state.items():
            clean[k.replace("module.", "")] = v
        model.load_state_dict(clean, strict=False)
        print(f"Loaded TransMorph from {path}")
    else:
        print(f"WARNING: TransMorph checkpoint not found at {path}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    ds = PairedImageFolderDataset(
        args.data_dir,
        flow_subdir="gt_flow",
        mask_subdir="valid_mask",
        img_size=tuple(args.img_size),
        grayscale=(args.input_nc == 1),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    print(f"Test samples: {len(ds)}")

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    if not args.no_pipeline:
        cut = load_cut(args.cut_ckpt, args.input_nc, args.output_nc, device)
        transmorph = load_transmorph(
            args.transmorph_ckpt, tuple(args.img_size),
            in_chans=args.output_nc * 2, device=device,
        )
        pipeline = PipelineInference(cut, transmorph).to(device)
    else:
        transmorph = load_transmorph(
            args.transmorph_ckpt, tuple(args.img_size),
            in_chans=args.input_nc * 2, device=device,
        )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    metrics = {
        "pre_zncc": AverageMeter(),
        "post_zncc": AverageMeter(),
        "pre_mse": AverageMeter(),
        "post_mse": AverageMeter(),
        "nmi": AverageMeter(),
        "fg_dice": AverageMeter(),
        "epe": AverageMeter(),
        "folding_pct": AverageMeter(),
    }

    all_results: list[dict] = []

    for idx, batch in enumerate(tqdm(loader, desc="Eval")):
        moving, fixed = batch[0].to(device), batch[1].to(device)
        gt_flow = batch[2].to(device) if len(batch) >= 3 else None
        valid_mask = batch[3].to(device) if len(batch) >= 4 else None

        # --- Run pipeline ---
        with torch.no_grad():
            if not args.no_pipeline:
                result = pipeline(moving, fixed)
                warped = result.warped
                flow = result.flow
                translated = result.translated
                # Intra-modal metrics (translated vs fixed)
                pre_zncc = _to_numpy_and_compute(moving, fixed, compute_zncc)
                post_zncc = _to_numpy_and_compute(translated, fixed, compute_zncc)
            else:
                x_in = torch.cat([fixed, moving], dim=1)
                warped, flow, _ = transmorph(x_in)
                pre_zncc = _to_numpy_and_compute(moving, fixed, compute_zncc)
                post_zncc = _to_numpy_and_compute(warped, fixed, compute_zncc)

        # --- Compute metrics ---
        moving_np = moving.cpu().numpy()[0, 0]
        fixed_np = fixed.cpu().numpy()[0, 0]
        warped_np = warped.cpu().numpy()[0, 0]
        flow_np = flow.cpu().numpy()[0]

        pre_mse_val = compute_mse(moving_np, fixed_np)
        post_mse_val = compute_mse(warped_np, fixed_np)
        nmi_val = compute_nmi(warped_np, fixed_np)
        fg_dice_val = compute_foreground_dice(warped_np, fixed_np)
        fold_val = compute_folding_ratio(flow_np)

        metrics["pre_zncc"].update(pre_zncc)
        metrics["post_zncc"].update(post_zncc)
        metrics["pre_mse"].update(pre_mse_val)
        metrics["post_mse"].update(post_mse_val)
        metrics["nmi"].update(nmi_val)
        metrics["fg_dice"].update(fg_dice_val)
        metrics["folding_pct"].update(fold_val)

        epe_val = 0.0
        if gt_flow is not None:
            epe_val = compute_epe(flow, gt_flow, valid_mask)
            metrics["epe"].update(epe_val)

        all_results.append({
            "idx": idx,
            "pre_zncc": pre_zncc,
            "post_zncc": post_zncc,
            "pre_mse": pre_mse_val,
            "post_mse": post_mse_val,
            "nmi": nmi_val,
            "fg_dice": fg_dice_val,
            "epe": epe_val,
            "folding_pct": fold_val,
        })

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results  (n={len(all_results)})")
    print(f"{'=' * 60}")
    for name, meter in metrics.items():
        if meter.count > 0:
            print(f"  {name:>12s}:  {meter.avg:.4f} ± {meter.std:.4f}")

    # Save to JSON
    summary = {
        "n_samples": len(all_results),
        "aggregates": {k: {"mean": m.avg, "std": m.std}
                       for k, m in metrics.items() if m.count > 0},
        "per_sample": all_results,
    }
    out_path = os.path.join(args.output_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


def _to_numpy_and_compute(
    a: torch.Tensor, b: torch.Tensor, fn,
) -> float:
    """Convert two (1,1,H,W) tensors to NumPy and call *fn*."""
    return float(fn(a.cpu().numpy()[0, 0], b.cpu().numpy()[0, 0]))


if __name__ == "__main__":
    main()
