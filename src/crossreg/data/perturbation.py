"""
Data perturbation utilities for cross-modality registration.

Generates synthetic deformations and appearance variations following the
SynthMorph / VoxelMorph data-augmentation strategy.

Typical usage::

    from crossreg.data.perturbation import (
        generate_diffeomorphic_flow,
        apply_synthmorph_appearance,
        center_crop_pair_and_flow,
    )

    flow, map_x, map_y = generate_diffeomorphic_flow((256, 256))
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import cv2
except ImportError:  # Pure non-rigid flow generation does not require OpenCV.
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise ImportError(
            "OpenCV is required for image remapping, affine perturbations, "
            "and perturbation dataset export"
        )
    return cv2

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: Default appearance-perturbation parameters (see :func:`apply_synthmorph_appearance`).
APPEARANCE_DEFAULTS: Dict[str, float] = {
    "gamma_prob": 0.8,
    "gamma_std": 0.3,
    "bias_prob": 0.8,
    "bias_res": 4,
    "bias_std": 0.4,
    "blur_prob": 0.5,
    "blur_sigma": 1.5,
    "noise_std": 0.05,
}

#: Minimum Jacobian determinant for diffeomorphism guarantee.
_JACOBIAN_EPS: float = 0.01

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_range(
    default: float,
    rng: Optional[Tuple[float, float]] = None,
) -> float:
    """Return *default* unless *rng* is given, then sample uniformly from it."""
    if rng is None:
        return float(default)
    lo, hi = rng
    if lo > hi:
        lo, hi = hi, lo
    return float(np.random.uniform(lo, hi))


def _jacobian_ok(dx: np.ndarray, dy: np.ndarray, eps: float = _JACOBIAN_EPS) -> bool:
    """Check whether the deformation (dx, dy) is fold-free."""
    dy_dy, dy_dx = np.gradient(dy)
    dx_dy, dx_dx = np.gradient(dx)
    jac = (1.0 + dx_dx) * (1.0 + dy_dy) - dx_dy * dy_dx
    return bool(np.min(jac) > eps)


def _disps_to_flow_and_maps(
    dx: np.ndarray,
    dy: np.ndarray,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (dx, dy) displacements to flow (2,H,W) + remap maps.

    Flow follows ``indexing='ij'``: ``flow[0]`` = dy, ``flow[1]`` = dx.
    """
    H, W = shape
    yc, xc = np.mgrid[0:H, 0:W]
    map_x = (xc.astype(np.float32) + dx.astype(np.float32)).astype(np.float32)
    map_y = (yc.astype(np.float32) + dy.astype(np.float32)).astype(np.float32)
    flow = np.stack([map_y - yc, map_x - xc], axis=0).astype(np.float32)
    return flow, map_x, map_y


def _random_border_mask(
    shape: Tuple[int, int],
    max_border_frac: float = 1 / 8,
    apply_prob: float = 0.5,
) -> np.ndarray:
    """Return all-ones mask, or (with *apply_prob*) one with random black borders.

    Border width is at most ``max_border_frac * dim`` on each side.
    """
    H, W = shape
    if np.random.rand() >= apply_prob:
        return np.ones((H, W), dtype=np.float32)

    def _rand_edge(dim: int) -> Tuple[int, int]:
        frac = max_border_frac
        a = int(np.random.choice([0, np.random.randint(0, max(1, int(dim * frac)))]))
        b = int(
            np.random.choice(
                [np.random.randint(max(1, int((1 - frac) * dim)), dim), dim]
            )
        )
        return a, b

    x0, x1 = _rand_edge(H)
    y0, y1 = _rand_edge(W)
    mask = np.zeros((H, W), dtype=np.float32)
    mask[x0:x1, y0:y1] = 1.0
    return mask


def _remap_warp(
    img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Wrap ``cv2.remap`` with linear interpolation and constant (0) border."""
    cv = _require_cv2()
    return cv.remap(
        img, map_x, map_y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )


def _make_valid_mask(
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Return a uint8 validity mask (255 = valid) for the given remap."""
    ones = np.ones_like(map_x, dtype=np.float32)
    cv = _require_cv2()
    valid = cv.remap(ones, map_x, map_y,
                      interpolation=cv.INTER_NEAREST,
                      borderMode=cv.BORDER_CONSTANT,
                      borderValue=0)
    return (valid * 255).astype(np.uint8)


def _tqdm_proxy(iterable: Sequence) -> Sequence:
    """tqdm wrapper; degrades gracefully if tqdm is not installed."""
    try:
        from tqdm import tqdm  # type: ignore[import-untyped]
        return tqdm(iterable)
    except ImportError:
        return iterable


def _read_image(path: str) -> np.ndarray:
    """Read a single-channel image, normalising non-uint8 to 0-255."""
    cv = _require_cv2()
    im = cv.imread(path, cv.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Failed to read: {path}")
    if im.dtype != np.uint8 and im.ndim == 2:
        im = cv.normalize(im, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return im


# ---------------------------------------------------------------------------
# Appearance perturbation
# ---------------------------------------------------------------------------


def apply_synthmorph_appearance(
    img: np.ndarray,
    config: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Apply stochastic appearance perturbations (SynthMorph-style).

    Applies, in order: gamma correction, low-frequency bias field,
    Gaussian blur, and additive Gaussian noise.  Each step is applied
    with an independent random probability.

    The input image is expected to be uint8 (0–255) single-channel.

    Parameters
    ----------
    img : np.ndarray
        Input image, shape (H, W), dtype uint8.
    config : dict, optional
        Keys: ``gamma_prob``, ``gamma_std``, ``bias_prob``, ``bias_res``,
        ``bias_std``, ``blur_prob``, ``blur_sigma``, ``noise_std``.

    Returns
    -------
    np.ndarray
        Perturbed image, same shape and dtype (uint8).
    """
    cfg = APPEARANCE_DEFAULTS if config is None else config
    x = img.astype(np.float32) / 255.0
    H, W = x.shape[:2]

    # 1. Gamma correction
    if np.random.rand() < cfg.get("gamma_prob", 0.8):
        gamma = float(np.exp(np.random.normal(0, cfg.get("gamma_std", 0.3))))
        x = np.power(x, gamma)

    # 2. Low-frequency multiplicative bias field
    if np.random.rand() < cfg.get("bias_prob", 0.8):
        cv = _require_cv2()
        res = int(cfg.get("bias_res", 4))
        grid = np.random.normal(0, cfg.get("bias_std", 0.4), (res, res))
        field = cv.resize(grid, (W, H), interpolation=cv.INTER_CUBIC)
        x = x * np.exp(field)

    # 3. Gaussian blur
    if np.random.rand() < cfg.get("blur_prob", 0.5):
        sigma = float(np.random.uniform(0, cfg.get("blur_sigma", 1.5)))
        x = gaussian_filter(x, sigma=sigma)

    # 4. Additive Gaussian noise
    x += np.random.normal(0, float(np.random.uniform(0, cfg.get("noise_std", 0.05))), x.shape)

    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Spatial / geometry perturbation
# ---------------------------------------------------------------------------


def set_random_zero_borders(
    im: np.ndarray,
    max_border_frac: float = 1 / 8,
    apply_prob: float = 0.5,
) -> np.ndarray:
    """Randomly zero out border regions (incomplete field-of-view simulation).

    Parameters
    ----------
    im : np.ndarray
        Input image, shape (H, W), any dtype.
    max_border_frac : float
        Maximum border width as fraction of each dimension (per side).
    apply_prob : float
        Probability of actually applying borders.

    Returns
    -------
    np.ndarray
        Image with (potentially) zeroed borders, same shape and dtype.
    """
    mask = _random_border_mask(im.shape[:2], max_border_frac, apply_prob)
    return (im.astype(np.float32) * mask).astype(im.dtype)


def sample_param_value(
    default: float,
    rng: Optional[Tuple[float, float]] = None,
) -> float:
    """Sample a value uniformly from *rng*, or return *default*."""
    return _sample_range(default, rng)


def generate_diffeomorphic_flow(
    shape: Tuple[int, int],
    smooth_sigma: float = 12.0,
    max_displacement: float = 15.0,
    affine_probability: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth, diffeomorphic deformation field with optional affine.

    Builds a non-rigid flow from smoothed Gaussian white noise (similar to
    Perlin noise).  The field is guaranteed fold-free via Jacobian check.

    Flow orientation is ``indexing='ij'``: ``flow[0]`` = dy, ``flow[1]`` = dx.

    Parameters
    ----------
    shape : (H, W)
        Spatial dimensions.
    smooth_sigma : float
        Gaussian sigma for smoothing the raw noise.  Larger → more global.
    max_displacement : float
        Maximum pixel displacement magnitude.
    affine_probability : float
        Probability of compounding a random affine (rotation ±5°, scale
        ±2 %, translation ±8 px).

    Returns
    -------
    flow : np.ndarray     shape (2, H, W), dtype float32
    map_x : np.ndarray    shape (H, W), for cv2.remap
    map_y : np.ndarray    shape (H, W), for cv2.remap
    """
    H, W = shape

    # 1. Fold-free smooth non-rigid displacement
    while True:
        dx = gaussian_filter(np.random.normal(0, 1, (H, W)).astype(np.float32),
                             sigma=smooth_sigma)
        dy = gaussian_filter(np.random.normal(0, 1, (H, W)).astype(np.float32),
                             sigma=smooth_sigma)

        mag = np.random.uniform(max_displacement * 0.3, max_displacement)
        dx *= mag / (np.max(np.abs(dx)) + 1e-5)
        dy *= mag / (np.max(np.abs(dy)) + 1e-5)

        if _jacobian_ok(dx, dy):
            break

    flow, map_x, map_y = _disps_to_flow_and_maps(dx, dy, shape)

    # 2. Optional affine compound
    if np.random.rand() < affine_probability:
        cv = _require_cv2()
        angle = float(np.random.uniform(-5, 5))
        scale = float(np.random.uniform(0.98, 1.02))
        tx, ty = np.random.uniform(-8, 8, 2).astype(float)
        M = cv.getRotationMatrix2D((W / 2, H / 2), angle, scale)

        mx = M[0, 0] * map_x + M[0, 1] * map_y + M[0, 2] + tx
        my = M[1, 0] * map_x + M[1, 1] * map_y + M[1, 2] + ty
        map_x, map_y = mx.astype(np.float32), my.astype(np.float32)

        yc, xc = np.mgrid[0:H, 0:W]
        flow = np.stack([map_y - yc, map_x - xc], axis=0).astype(np.float32)

    return flow, map_x, map_y


def generate_ffd_like_flow(
    shape: Tuple[int, int],
    grid_resolution: int = 8,
    max_displacement: float = 15.0,
    smooth_sigma: float = 1.0,
    max_retries: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate an FFD (Free-Form Deformation) style flow field.

    Samples random displacements on a coarse ``grid_resolution × grid_resolution``
    control-point lattice, up-samples to full resolution, and optionally smooths.

    Falls back to :func:`generate_diffeomorphic_flow` if a fold-free field
    cannot be produced within *max_retries*.

    Parameters
    ----------
    shape : (H, W)
        Spatial dimensions.
    grid_resolution : int
        Control-point count per side.
    max_displacement : float
        Maximum displacement at control points.
    smooth_sigma : float
        Post-interpolation Gaussian sigma (0 = no smoothing).
    max_retries : int
        Attempts before falling back to diffeomorphic flow.

    Returns
    -------
    flow : np.ndarray     shape (2, H, W), dtype float32
    map_x : np.ndarray    shape (H, W)
    map_y : np.ndarray    shape (H, W)
    """
    H, W = shape
    cv = _require_cv2()

    for _ in range(max_retries):
        dx_g = np.random.uniform(-max_displacement, max_displacement,
                                 (grid_resolution, grid_resolution)).astype(np.float32)
        dy_g = np.random.uniform(-max_displacement, max_displacement,
                                 (grid_resolution, grid_resolution)).astype(np.float32)

        dx = cv.resize(dx_g, (W, H), interpolation=cv.INTER_CUBIC)
        dy = cv.resize(dy_g, (W, H), interpolation=cv.INTER_CUBIC)

        if smooth_sigma > 0:
            dx = gaussian_filter(dx, sigma=smooth_sigma)
            dy = gaussian_filter(dy, sigma=smooth_sigma)

        if _jacobian_ok(dx, dy):
            return _disps_to_flow_and_maps(dx, dy, shape)

    # Fallback
    return generate_diffeomorphic_flow(shape, smooth_sigma=12.0,
                                       max_displacement=max_displacement,
                                       affine_probability=0.0)


# ---------------------------------------------------------------------------
# Spatial-consistency helpers
# ---------------------------------------------------------------------------


def center_crop_pair_and_flow(
    moving: np.ndarray,
    fixed: np.ndarray,
    valid_mask: np.ndarray,
    flow: np.ndarray,
    keep_fraction: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center-crop an image pair, mask, and flow field consistently.

    Parameters
    ----------
    moving, fixed : np.ndarray    Images, shape (H, W) or (H, W, C).
    valid_mask : np.ndarray       shape (H, W).
    flow : np.ndarray             shape (2, H, W).
    keep_fraction : float         Fraction of original size to retain, in (0, 1].

    Returns
    -------
    moving_c, fixed_c, mask_c, flow_c
        Cropped versions of the inputs.

    Raises
    ------
    ValueError
        If shapes are inconsistent or *keep_fraction* is out of bounds.
    """
    if not (0 < keep_fraction <= 1):
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")

    H, W = moving.shape[:2]
    if fixed.shape[:2] != (H, W) or valid_mask.shape[:2] != (H, W):
        raise ValueError("moving, fixed, valid_mask must share (H, W)")
    if flow.ndim != 3 or flow.shape[0] != 2 or flow.shape[1:] != (H, W):
        raise ValueError(f"flow must be (2, H, W), got {flow.shape}")

    ch, cw = max(1, int(round(H * keep_fraction))), max(1, int(round(W * keep_fraction)))
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    y1, x1 = y0 + ch, x0 + cw

    return moving[y0:y1, x0:x1], fixed[y0:y1, x0:x1], \
        valid_mask[y0:y1, x0:x1], flow[:, y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Multi-modal test-suite builder
# ---------------------------------------------------------------------------


# Default suite configs (for when no user config is supplied).
_SUITE_DEFAULTS: List[Dict[str, Any]] = [
    {
        "name": "test_id",
        "warp_mode": "diffeomorphic",
        "smooth_sigma": 12.0,
        "max_displacement": 15.0,
        "affine_probability": 0.8,
        "add_appearance": False,
    },
    {
        "name": "test_shift",
        "warp_mode": "diffeomorphic",
        "smooth_sigma": 12.0,
        "smooth_sigma_range": (7.0, 10.0),
        "max_displacement": 15.0,
        "max_displacement_range": (17.0, 21.0),
        "affine_probability": 0.8,
        "add_appearance": True,
        "appearance_config": {
            "gamma_prob": 0.9, "gamma_std": 0.45,
            "bias_prob": 0.9, "bias_res": 3, "bias_std": 0.6,
            "blur_prob": 0.7, "blur_sigma": 2.2,
            "noise_std": 0.08,
        },
    },
    {
        "name": "test_ffd",
        "warp_mode": "ffd",
        "ffd_grid_resolution": 8,
        "ffd_smooth_sigma": 1.0,
        "max_displacement": 15.0,
        "add_appearance": False,
    },
]


def _resolve_flow(
    shape: Tuple[int, int],
    suite_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the appropriate flow generator based on suite config."""
    max_disp = _sample_range(
        suite_cfg.get("max_displacement", 15.0),
        suite_cfg.get("max_displacement_range"),
    )

    if suite_cfg.get("warp_mode", "diffeomorphic") == "ffd":
        return generate_ffd_like_flow(
            shape,
            grid_resolution=suite_cfg.get("ffd_grid_resolution", 8),
            max_displacement=max_disp,
            smooth_sigma=suite_cfg.get("ffd_smooth_sigma", 1.0),
        )

    smooth = _sample_range(
        suite_cfg.get("smooth_sigma", 12.0),
        suite_cfg.get("smooth_sigma_range"),
    )
    return generate_diffeomorphic_flow(
        shape,
        smooth_sigma=smooth,
        max_displacement=max_disp,
        affine_probability=suite_cfg.get("affine_probability", 0.8),
    )


def _save_sample(
    task_dirs: Dict[str, Dict[str, str]],
    moving: np.ndarray,
    fixed: np.ndarray,
    mask: np.ndarray,
    flow: np.ndarray,
    file_prefix: str,
) -> None:
    """Write moving, fixed, mask (PNG) and flow (npy) to disk."""
    cv = _require_cv2()
    cv.imwrite(os.path.join(task_dirs["moving"], f"{file_prefix}.png"),
                moving.astype(np.uint8))
    cv.imwrite(os.path.join(task_dirs["fixed"], f"{file_prefix}.png"),
                fixed.astype(np.uint8))
    cv.imwrite(os.path.join(task_dirs["mask"], f"{file_prefix}.png"),
                mask.astype(np.uint8))
    np.save(os.path.join(task_dirs["flow"], f"{file_prefix}.npy"), flow)


def build_multimodal_test_suites(
    modal_dirs: Dict[str, str],
    task_pairs: List[Tuple[str, str]],
    output_dir: str = "./Train_CrossModal",
    random_seed: int = 42,
    suite_configs: Optional[List[Dict[str, Any]]] = None,
    use_zero_borders: bool = True,
    use_no_edge_cue: bool = False,
    no_edge_cue_keep_fraction: float = 0.85,
) -> None:
    """Construct cross-modal test suites with synthetic deformations.

    Given per-modality directories containing filename-aligned images,
    this generates for each ``(moving_mod, fixed_mod)`` pair:

    - A shared random deformation field.
    - Warped versions of each modality.
    - Paired outputs: moving, fixed, gt_flow, valid_mask.

    Multiple suites each get their own ``output_dir/<suite>/`` subtree.

    Output layout::

        <output_dir>/<suite>/<src>_to_<tgt>/
            fixed/   moving/   gt_flow/   valid_mask/

    Parameters
    ----------
    modal_dirs : dict
        ``{modality_name: image_folder}``.
    task_pairs : list of (str, str)
        Directional pairs, e.g. ``[("ch0","ch1"), ("ch1","ch0")]``.
    output_dir : str
        Root output path.
    random_seed : int
        Seed for reproducible shuffling.
    suite_configs : list of dict, optional
        Suite descriptors.  Keys:
        ``name``, ``warp_mode`` (diffeomorphic|ffd),
        ``smooth_sigma`` / ``smooth_sigma_range``,
        ``max_displacement`` / ``max_displacement_range``,
        ``affine_probability``,
        ``ffd_grid_resolution``, ``ffd_smooth_sigma``,
        ``add_appearance``, ``appearance_config``,
        ``use_no_edge_cue``, ``no_edge_cue_keep_fraction``.
    use_zero_borders : bool
        Global toggle for random black-border simulation.
    use_no_edge_cue : bool
        Global toggle for center-crop edge-cue removal.
    no_edge_cue_keep_fraction : float
        Global keep fraction for edge-cue removal.
    """
    # --- Index images by filename ---
    names = list(modal_dirs)
    files_by_mod: Dict[str, Dict[str, str]] = {}
    for mod, d in modal_dirs.items():
        paths = glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.tif*"))
        files_by_mod[mod] = {os.path.basename(p): p for p in paths}

    common = sorted(set(files_by_mod[names[0]]))
    for mod in names[1:]:
        common = sorted(set(common) & set(files_by_mod[mod]))
    if not common:
        print("No paired images found — check input directories.")
        return

    print(f"Found {len(common)} aligned images across all modalities.")
    rng = np.random.default_rng(random_seed)
    common = np.array(rng.permutation(common))

    if suite_configs is None:
        suite_configs = _SUITE_DEFAULTS

    print(f"Tasks: {[' → '.join(p) for p in task_pairs]}")
    print(f"Suites: {[c['name'] for c in suite_configs]}")

    single = len(suite_configs) == 1

    for cfg in suite_configs:
        suite_root = output_dir if single else os.path.join(output_dir, cfg["name"])
        print(f"\n===== Suite: {cfg['name']} ({len(common)} base sets) =====")

        # Suite-level overrides
        edge_cue = cfg.get("use_no_edge_cue", use_no_edge_cue)
        keep = cfg.get("no_edge_cue_keep_fraction", no_edge_cue_keep_fraction)
        add_app = cfg.get("add_appearance", False)
        app_cfg = cfg.get("appearance_config", APPEARANCE_DEFAULTS)

        if edge_cue:
            print(f"[no_edge_cue] enabled, keep_fraction={keep}")

        # Pre-create output directories for all task pairs
        task_dirs: Dict[str, Dict[str, str]] = {}
        for src, tgt in task_pairs:
            tn = f"{src}_to_{tgt}"
            task_dirs[tn] = {
                sub: os.path.join(suite_root, tn, sub)
                for sub in ("fixed", "moving", "gt_flow", "valid_mask")
            }
            for d in task_dirs[tn].values():
                os.makedirs(d, exist_ok=True)

        for idx, key in enumerate(_tqdm_proxy(common)):
            # Load aligned images
            imgs: Dict[str, np.ndarray] = {}
            shape: Optional[Tuple[int, int]] = None
            for mod in names:
                im = _read_image(files_by_mod[mod][key])
                imgs[mod] = im
                shape = im.shape[:2]
            H, W = shape

            # Shared border mask
            border_mask = _random_border_mask((H, W),
                                              apply_prob=0.5 if use_zero_borders else 0.0)
            base = {m: (imgs[m].astype(np.float32) * border_mask).astype(imgs[m].dtype)
                    for m in names}

            # Shared deformation
            flow, map_x, map_y = _resolve_flow((H, W), cfg)
            valid_mask = _make_valid_mask(map_x, map_y)

            # Warp all modalities
            warped = {m: _remap_warp(base[m], map_x, map_y) for m in names}

            prefix = f"{idx:04d}_{key.rsplit('.', 1)[0]}"

            for src, tgt in task_pairs:
                tn = f"{src}_to_{tgt}"
                mov, fix = base[src], warped[tgt]

                # Appearance perturbation
                if add_app:
                    mov = apply_synthmorph_appearance(mov, app_cfg)
                    fix = apply_synthmorph_appearance(fix, app_cfg)

                # Edge-cue removal
                t_mask, t_flow = valid_mask, flow
                if edge_cue:
                    mov, fix, t_mask, t_flow = center_crop_pair_and_flow(
                        mov, fix, valid_mask, flow, keep_fraction=keep,
                    )

                _save_sample(task_dirs[tn], mov, fix, t_mask, t_flow, prefix)
