"""Datasets for paired and unpaired image-to-image translation."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _images(directory: str) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"image directory not found: {directory}")
    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise RuntimeError(f"no supported images found in {directory}")
    return paths


def _by_stem(paths: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        if path.stem in result:
            raise RuntimeError(f"duplicate image stem '{path.stem}' in {path.parent}")
        result[path.stem] = path
    return result


def _transform_pair(
    images: list[Image.Image], load_size: int, crop_size: int,
    channels: list[int], shared_geometry: bool,
) -> list[torch.Tensor]:
    if crop_size > load_size:
        raise ValueError("crop_size must be <= load_size")
    resized = [
        TF.resize(image, [load_size, load_size], interpolation=InterpolationMode.BICUBIC)
        for image in images
    ]
    shared_flip = random.random() < 0.5 if shared_geometry else None
    shared_crop = T.RandomCrop.get_params(
        resized[0], output_size=(crop_size, crop_size)
    ) if shared_geometry else None
    outputs = []
    for image, num_channels in zip(resized, channels):
        do_flip = shared_flip if shared_geometry else random.random() < 0.5
        if do_flip:
            image = TF.hflip(image)
        crop = shared_crop or T.RandomCrop.get_params(
            image, output_size=(crop_size, crop_size)
        )
        image = TF.crop(image, *crop)
        tensor = TF.to_tensor(image)
        outputs.append(TF.normalize(tensor, [0.5] * num_channels, [0.5] * num_channels))
    return outputs


class TwoDomainTranslationDataset(Dataset):
    """Two-domain translation data with explicit paired/unpaired semantics.

    ``unpaired`` is the CUT default: all images are used, domains are sampled
    independently, and geometric augmentation is independent. ``paired``
    matches filename stems and applies shared geometry.
    """

    def __init__(
        self, dir_a: str, dir_b: str, input_nc: int = 1, output_nc: int = 1,
        load_size: int = 286, crop_size: int = 256,
        pairing_mode: str = "unpaired",
    ):
        if pairing_mode not in {"unpaired", "paired"}:
            raise ValueError("pairing_mode must be 'unpaired' or 'paired'")
        self.paths_a = _images(dir_a)
        self.paths_b = _images(dir_b)
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.load_size = load_size
        self.crop_size = crop_size
        self.pairing_mode = pairing_mode

        self.pairs: list[tuple[Path, Path]] = []
        if pairing_mode == "paired":
            by_a, by_b = _by_stem(self.paths_a), _by_stem(self.paths_b)
            common = sorted(by_a.keys() & by_b.keys())
            if not common:
                raise RuntimeError(f"no matching image stems found in {dir_a} and {dir_b}")
            self.pairs = [(by_a[key], by_b[key]) for key in common]

    def __len__(self) -> int:
        return len(self.pairs) if self.pairing_mode == "paired" else max(len(self.paths_a), len(self.paths_b))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if self.pairing_mode == "paired":
            path_a, path_b = self.pairs[index]
        else:
            path_a = self.paths_a[index % len(self.paths_a)]
            path_b = random.choice(self.paths_b)
        image_a = Image.open(path_a).convert("L" if self.input_nc == 1 else "RGB")
        image_b = Image.open(path_b).convert("L" if self.output_nc == 1 else "RGB")
        tensor_a, tensor_b = _transform_pair(
            [image_a, image_b], self.load_size, self.crop_size,
            [self.input_nc, self.output_nc],
            shared_geometry=self.pairing_mode == "paired",
        )
        return {
            "A": tensor_a, "B": tensor_b,
            "A_paths": str(path_a), "B_paths": str(path_b),
        }


class MultiDomainTranslationDataset(Dataset):
    """N-domain translation data with unpaired or common-stem paired sampling."""

    def __init__(
        self, modality_dirs: list[str], input_nc: int = 1,
        load_size: int = 286, crop_size: int = 256,
        pairing_mode: str = "unpaired",
        modality_files: list[list[str | Path]] | None = None,
        paired_anchor_files: list[list[str | Path]] | None = None,
        paired_anchor_probability: float = 0.0,
    ):
        if len(modality_dirs) < 2:
            raise ValueError("at least two modality directories are required")
        if pairing_mode not in {"unpaired", "paired"}:
            raise ValueError("pairing_mode must be 'unpaired' or 'paired'")
        if not 0.0 <= paired_anchor_probability <= 1.0:
            raise ValueError("paired_anchor_probability must be in [0, 1]")
        if paired_anchor_probability > 0.0 and paired_anchor_files is None:
            raise ValueError(
                "paired_anchor_files are required when anchor probability is positive"
            )
        if pairing_mode == "paired" and paired_anchor_probability > 0.0:
            raise ValueError(
                "paired anchors are only needed when pairing_mode is unpaired"
            )
        if modality_files is not None:
            if len(modality_files) != len(modality_dirs):
                raise ValueError("modality_files must match modality_dirs length")
            self.paths = []
            for directory, files in zip(modality_dirs, modality_files):
                root = Path(directory).resolve()
                paths = sorted(Path(path).absolute() for path in files)
                if not paths:
                    raise RuntimeError(f"no selected images found in {directory}")
                invalid = [
                    path for path in paths
                    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS
                    or path.parent.resolve() != root
                ]
                if invalid:
                    raise RuntimeError(f"invalid selected image for {directory}: {invalid[0]}")
                self.paths.append(paths)
        else:
            self.paths = [_images(directory) for directory in modality_dirs]
        self.input_nc = input_nc
        self.load_size = load_size
        self.crop_size = crop_size
        self.pairing_mode = pairing_mode
        self.num_modalities = len(self.paths)
        self.paired_anchor_probability = paired_anchor_probability
        self.paired_paths: list[dict[str, Path]] = []
        self.common_stems: list[str] = []
        if pairing_mode == "paired":
            self.paired_paths = [_by_stem(paths) for paths in self.paths]
            common = set(self.paired_paths[0])
            for mapping in self.paired_paths[1:]:
                common &= mapping.keys()
            self.common_stems = sorted(common)
            if not self.common_stems:
                raise RuntimeError("no filename stem is shared by all modality directories")

        self.paired_anchor_paths: list[dict[str, Path]] = []
        self.paired_anchor_stems: list[str] = []
        if paired_anchor_files is not None:
            if len(paired_anchor_files) != len(modality_dirs):
                raise ValueError("paired_anchor_files must match modality_dirs length")
            validated_anchor_paths = []
            for directory, files in zip(modality_dirs, paired_anchor_files):
                root = Path(directory).resolve()
                paths = sorted(Path(path).absolute() for path in files)
                invalid = [
                    path for path in paths
                    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS
                    or path.parent.resolve() != root
                ]
                if invalid:
                    raise RuntimeError(
                        f"invalid paired anchor for {directory}: {invalid[0]}"
                    )
                validated_anchor_paths.append(paths)
            self.paired_anchor_paths = [
                _by_stem(paths) for paths in validated_anchor_paths
            ]
            common = set(self.paired_anchor_paths[0])
            for mapping in self.paired_anchor_paths[1:]:
                common &= mapping.keys()
            self.paired_anchor_stems = sorted(common)
            if not self.paired_anchor_stems:
                raise RuntimeError(
                    "no filename stem is shared by all paired anchor domains"
                )

    def __len__(self) -> int:
        if self.pairing_mode == "paired":
            return len(self.common_stems) * self.num_modalities
        return max(len(paths) for paths in self.paths) * self.num_modalities

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        src_id = random.randrange(self.num_modalities)
        tgt_id = random.randrange(self.num_modalities - 1)
        if tgt_id >= src_id:
            tgt_id += 1

        is_paired = self.pairing_mode == "paired"
        use_anchor = (
            self.pairing_mode == "unpaired"
            and self.paired_anchor_probability > 0.0
            and random.random() < self.paired_anchor_probability
        )
        if is_paired:
            stem = self.common_stems[index % len(self.common_stems)]
            src_path = self.paired_paths[src_id][stem]
            tgt_path = self.paired_paths[tgt_id][stem]
        elif use_anchor:
            stem = random.choice(self.paired_anchor_stems)
            src_path = self.paired_anchor_paths[src_id][stem]
            tgt_path = self.paired_anchor_paths[tgt_id][stem]
            is_paired = True
        else:
            src_path = random.choice(self.paths[src_id])
            tgt_path = random.choice(self.paths[tgt_id])

        mode = "L" if self.input_nc == 1 else "RGB"
        source = Image.open(src_path).convert(mode)
        target = Image.open(tgt_path).convert(mode)
        source_t, target_t = _transform_pair(
            [source, target], self.load_size, self.crop_size,
            [self.input_nc, self.input_nc],
            shared_geometry=is_paired,
        )
        return {
            "A": source_t, "B": target_t,
            "src_id": src_id, "tgt_id": tgt_id,
            "is_paired": is_paired,
            "A_paths": str(src_path), "B_paths": str(tgt_path),
        }
