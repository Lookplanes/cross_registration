import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import pkload


def _apply_seeded_subset(pairs, sample_count=None, sample_seed=0):
    """Select a stable random subset from already-sorted pairs.

    Args:
        pairs: list of pair dicts.
        sample_count: number of samples to keep. None means keep all.
        sample_seed: seed controlling subset choice.
    """
    if sample_count is None:
        return pairs

    sample_count = int(sample_count)
    if sample_count <= 0:
        return []

    total = len(pairs)
    if sample_count >= total:
        return pairs

    rng = random.Random(int(sample_seed))
    chosen = set(rng.sample(range(total), sample_count))
    # Keep original sorted order for readability and reproducibility.
    return [p for idx, p in enumerate(pairs) if idx in chosen]


class RaFDDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_gray, y_gray = pkload(path)
        x_gray, y_gray = x_gray[None, ...], y_gray[None, ...]
        x_gray, y_gray = self.transforms([x_gray, y_gray])
        x = np.ascontiguousarray(x_gray)
        y = np.ascontiguousarray(y_gray)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class RaFDInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_gray, y_gray = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_gray, y_gray = x_gray[None, ...], y_gray[None, ...]
        x_gray = np.ascontiguousarray(x_gray.astype(np.float32))
        y_gray = np.ascontiguousarray(y_gray.astype(np.float32))
        x = np.ascontiguousarray(x.astype(np.float32))
        y = np.ascontiguousarray(y.astype(np.float32))
        x_gray, y_gray = torch.from_numpy(x_gray), torch.from_numpy(y_gray)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y, x_gray, y_gray

    def __len__(self):
        return len(self.paths)


class PairedImageFolderDataset(Dataset):
    """Paired PNG/JPG dataset for 2D registration.

    Directory layout (recommended):
      root/
        moving/  (source domain)
        fixed/   (target domain)
        mask/    (optional binary/tissue mask aligned with fixed)

    Pairing strategy: filename match by stem (e.g. 001.png in both folders).

    Returns:
      moving, fixed, (optional) fixed_mask
      shapes:
        moving: [3,H,W] float32 in [0,1]
        fixed:  [3,H,W] float32 in [0,1]
        mask:   [1,H,W] float32 in {0,1}
    """

    def __init__(
        self,
        root_dir: str,
        moving_subdir: str = 'moving',
        fixed_subdir: str = 'fixed',
        mask_subdir: str | None = None,
        flow_subdir: str | None = None,
        img_size: tuple[int, int] | None = None,
        transforms=None,
        grayscale: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.moving_dir = os.path.join(root_dir, moving_subdir)
        self.fixed_dir = os.path.join(root_dir, fixed_subdir)
        self.mask_dir = os.path.join(root_dir, mask_subdir) if mask_subdir else None
        self.flow_dir = os.path.join(root_dir, flow_subdir) if flow_subdir else None
        self.img_size = img_size
        self.transforms = transforms
        self.grayscale = grayscale

        if not os.path.isdir(self.moving_dir):
            raise FileNotFoundError(f"moving dir not found: {self.moving_dir}")
        if not os.path.isdir(self.fixed_dir):
            raise FileNotFoundError(f"fixed dir not found: {self.fixed_dir}")

        exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        moving_files = [f for f in os.listdir(self.moving_dir) if f.lower().endswith(exts)]
        fixed_files = set([f for f in os.listdir(self.fixed_dir) if f.lower().endswith(exts)])
        self.pairs = []
        for f in moving_files:
            if f in fixed_files:
                self.pairs.append((os.path.join(self.moving_dir, f), os.path.join(self.fixed_dir, f)))
        self.pairs.sort()

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No paired images found. Expected matching filenames under {self.moving_dir} and {self.fixed_dir}"
            )

    def _read_rgb(self, path: str) -> np.ndarray:
        from PIL import Image

        mode = "L" if self.grayscale else "RGB"
        img = Image.open(path).convert(mode)
        if self.img_size is not None:
            img = img.resize((self.img_size[1], self.img_size[0]), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if not self.grayscale:
            arr = arr.transpose(2, 0, 1)  # [3, H, W]
        else:
            arr = arr[None, ...]  # [1, H, W]
        return arr

    def _read_mask(self, path: str) -> np.ndarray:
        from PIL import Image

        img = Image.open(path).convert('L')
        if self.img_size is not None:
            img = img.resize((self.img_size[1], self.img_size[0]), resample=Image.NEAREST)
        arr = (np.asarray(img, dtype=np.float32) > 0).astype(np.float32)  # [H,W]
        return arr[None, ...]  # [1,H,W]

    def __getitem__(self, index):
        moving_path, fixed_path = self.pairs[index]
        moving = self._read_rgb(moving_path)
        fixed = self._read_rgb(fixed_path)

        if self.transforms is not None:
            moving, fixed = self.transforms([moving, fixed])

        moving = np.ascontiguousarray(moving.astype(np.float32))
        fixed = np.ascontiguousarray(fixed.astype(np.float32))

        moving_t = torch.from_numpy(moving)
        fixed_t = torch.from_numpy(fixed)

        flow_t = None
        if self.flow_dir is not None:
            # Match flow field by filename stem (e.g., 001.png -> 001.npy)
            stem = os.path.splitext(os.path.basename(fixed_path))[0]
            flow_path = os.path.join(self.flow_dir, stem + '.npy')
            if os.path.isfile(flow_path):
                flow = np.load(flow_path)
                flow_t = torch.from_numpy(np.ascontiguousarray(flow.astype(np.float32)))

        if self.mask_dir is None:
            if flow_t is not None:
                return moving_t, fixed_t, flow_t
            return moving_t, fixed_t

        mask_path = os.path.join(self.mask_dir, os.path.basename(fixed_path))
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"mask not found for {fixed_path}: {mask_path}")
        mask = self._read_mask(mask_path)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))
        if flow_t is not None:
            return moving_t, fixed_t, mask_t, flow_t
        return moving_t, fixed_t, mask_t

    def __len__(self):
        return len(self.pairs)


class MultiModalityPairedDataset(Dataset):
    """
    Multi-modality dataset for 2D intra-modality registration.
    Expects subfolders containing 'moving', 'fixed', and 'gt_flow' (optional).
    """
    def __init__(
        self,
        root_dir: str,
        img_size: tuple[int, int] | None = None,
        transforms=None,
        selected_names: list[str] | None = None,
        sample_count: int | None = None,
        sample_seed: int = 0,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.transforms = transforms
        self.pairs = []
        selected_name_set = set(selected_names) if selected_names else None

        # Walk subfolders under root_dir (e.g., Modality_X, Modality_Y).
        for modality_folder in os.listdir(root_dir):
            modality_path = os.path.join(root_dir, modality_folder)
            if not os.path.isdir(modality_path):
                continue

            moving_dir = os.path.join(modality_path, 'moving')
            fixed_dir = os.path.join(modality_path, 'fixed')
            flow_dir = os.path.join(modality_path, 'gt_flow')
            mask_dir = os.path.join(modality_path, 'valid_mask')

            if os.path.isdir(moving_dir) and os.path.isdir(fixed_dir):
                exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
                moving_files = [f for f in os.listdir(moving_dir) if f.lower().endswith(exts)]
                fixed_files = set([f for f in os.listdir(fixed_dir) if f.lower().endswith(exts)])

                for f in moving_files:
                    if f in fixed_files:
                        if selected_name_set is not None and f not in selected_name_set:
                            continue
                        flow_path = None
                        if os.path.isdir(flow_dir):
                            stem = os.path.splitext(f)[0]
                            possible_flow = os.path.join(flow_dir, stem + '.npy')
                            if os.path.isfile(possible_flow):
                                flow_path = possible_flow

                        mask_path = None
                        if os.path.isdir(mask_dir):
                            possible_mask = os.path.join(mask_dir, f)
                            if os.path.isfile(possible_mask):
                                mask_path = possible_mask

                        self.pairs.append({
                            'moving': os.path.join(moving_dir, f),
                            'fixed': os.path.join(fixed_dir, f),
                            'flow': flow_path,
                            'mask': mask_path
                        })

        # Stable ordering for reproducible sampling across runs/baselines.
        self.pairs.sort(key=lambda p: (p['moving'], p['fixed']))
        self.pairs = _apply_seeded_subset(self.pairs, sample_count=sample_count, sample_seed=sample_seed)

        if len(self.pairs) == 0:
            print(f"Warning: No paired images found under subdirectories of {self.root_dir}")

    def _read_mask(self, path: str) -> np.ndarray:
        from PIL import Image
        img = Image.open(path).convert('L')
        if self.img_size is not None:
            img = img.resize((self.img_size[1], self.img_size[0]), resample=Image.NEAREST)
        arr = (np.asarray(img, dtype=np.float32) > 0).astype(np.float32)  # [H, W]
        return arr[None, ...]  # [1, H, W]

    def _read_rgb_and_normalize(self, path: str) -> np.ndarray:
        from PIL import Image
        # Use L mode for grayscale images.
        img = Image.open(path).convert('L')
        if self.img_size is not None:
            img = img.resize((self.img_size[1], self.img_size[0]), resample=Image.BILINEAR)

        arr = np.asarray(img, dtype=np.float32)

        # Min-max normalization to [0, 1].
        if arr.max() > 1.0:
            arr = arr / 255.0

        arr = arr[None, ...]
        return arr

    def __getitem__(self, index):
        pair = self.pairs[index]
        moving = self._read_rgb_and_normalize(pair['moving'])
        fixed = self._read_rgb_and_normalize(pair['fixed'])

        mask = None
        if pair.get('mask') is not None:
            mask = self._read_mask(pair['mask'])

        flow = None
        if pair['flow'] is not None:
            flow = np.load(pair['flow']).astype(np.float32)
            if flow.ndim == 3:
                # Ensure shape is (C, H, W). If (H, W, 2), transpose to (2, H, W).
                if flow.shape[2] == 2:
                    flow = flow.transpose(2, 0, 1)

        if self.transforms is not None:
            if flow is not None:
                # Pass flow to transforms (k=2 logic handled in RandomFlip).
                out = self.transforms([moving, fixed, flow])
                moving, fixed, flow = out[0], out[1], out[2]
            else:
                out = self.transforms([moving, fixed])
                moving, fixed = out[0], out[1]

        moving = np.ascontiguousarray(moving.astype(np.float32))
        fixed = np.ascontiguousarray(fixed.astype(np.float32))

        moving_t = torch.from_numpy(moving)
        fixed_t = torch.from_numpy(fixed)

        flow_t = None
        if flow is not None:
            flow_t = torch.from_numpy(np.ascontiguousarray(flow.astype(np.float32)))

        mask_t = None
        if mask is not None:
            mask_t = torch.from_numpy(np.ascontiguousarray(mask))

        if flow_t is not None and mask_t is not None:
            return moving_t, fixed_t, flow_t, mask_t
        elif flow_t is not None:
            return moving_t, fixed_t, flow_t
        elif mask_t is not None:
            return moving_t, fixed_t, None, mask_t

        return moving_t, fixed_t

    def __len__(self):
        return len(self.pairs)


class SingleModalityPairedDataset(MultiModalityPairedDataset):
    """Single-modality version of MultiModalityPairedDataset.

    Only reads a specified directory instead of traversing all subfolders.
    """
    def __init__(
        self,
        target_dir: str,
        moving_folder: str = 'moving',
        fixed_folder: str = 'fixed',
        img_size: tuple = None,
        transforms=None,
        selected_names: list[str] | None = None,
        sample_count: int | None = None,
        sample_seed: int = 0,
    ):
        torch.utils.data.Dataset.__init__(self)
        self.root_dir = target_dir
        self.img_size = img_size
        self.transforms = transforms
        self.pairs = []
        selected_name_set = set(selected_names) if selected_names else None

        moving_dir = os.path.join(target_dir, moving_folder)
        fixed_dir = os.path.join(target_dir, fixed_folder)
        flow_dir = os.path.join(target_dir, 'gt_flow')
        mask_dir = os.path.join(target_dir, 'valid_mask')

        if os.path.isdir(moving_dir) and os.path.isdir(fixed_dir):
            exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
            moving_files = [f for f in os.listdir(moving_dir) if f.lower().endswith(exts)]
            fixed_files = set([f for f in os.listdir(fixed_dir) if f.lower().endswith(exts)])

            for f in moving_files:
                if f in fixed_files:
                    if selected_name_set is not None and f not in selected_name_set:
                        continue
                    flow_path = None
                    if os.path.isdir(flow_dir):
                        stem = os.path.splitext(f)[0]
                        possible_flow = os.path.join(flow_dir, stem + '.npy')
                        if os.path.isfile(possible_flow):
                            flow_path = possible_flow

                    mask_path = None
                    if os.path.isdir(mask_dir):
                        possible_mask = os.path.join(mask_dir, f)
                        if os.path.isfile(possible_mask):
                            mask_path = possible_mask

                    self.pairs.append({
                        'moving': os.path.join(moving_dir, f),
                        'fixed': os.path.join(fixed_dir, f),
                        'flow': flow_path,
                        'mask': mask_path
                    })

        # Stable ordering for reproducible sampling across runs/baselines.
        self.pairs.sort(key=lambda p: (p['moving'], p['fixed']))
        self.pairs = _apply_seeded_subset(self.pairs, sample_count=sample_count, sample_seed=sample_seed)
