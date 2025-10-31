from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils import data
from torchvision.transforms.functional import normalize

from neosr.data.transforms import basic_augment, paired_random_crop
from neosr.utils import get_root_logger, img2tensor
from neosr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CachedSRDataset(data.Dataset):
    """Paired SR dataset that keeps image pairs in host RAM for fast sampling."""

    def __init__(self, opt: dict[str, Any]) -> None:
        super().__init__()
        self.opt = opt
        self.logger = get_root_logger()
        self.scale = opt.get("scale", 1)
        self.patch_size = opt.get("patch_size")
        self.phase = opt.get("phase", "train")
        self.mean = opt.get("mean")
        self.std = opt.get("std")
        self.color = opt.get("color", None) != "y"
        self.use_hflip = opt.get("use_hflip", True)
        self.use_rot = opt.get("use_rot", True)

        cached_lq = opt.get("_cached_lq_images")
        cached_gt = opt.get("_cached_gt_images")
        cached_lq_paths = opt.get("_cached_lq_paths")
        cached_gt_paths = opt.get("_cached_gt_paths")
        cached_rel = opt.get("_cached_rel_paths")

        self.cache_in_ram_active = False
        if cached_lq and cached_gt:
            self.lq_images = cached_lq
            self.gt_images = cached_gt
            self.lq_paths = cached_lq_paths or [
                f"{opt.get('dataroot_lq', 'lq')}::{idx}" for idx in range(len(self.lq_images))
            ]
            self.gt_paths = cached_gt_paths or [
                f"{opt.get('dataroot_gt', 'gt')}::{idx}" for idx in range(len(self.gt_images))
            ]
            self.rel_paths = cached_rel or [str(idx) for idx in range(len(self.lq_images))]
            self.cache_in_ram_active = True
            self.logger.info(
                "CachedSRDataset reusing %d preloaded pairs from build_dataset cache.",
                len(self.lq_images),
            )
        else:
            self.logger.info("CachedSRDataset preloading images locally (no shared cache provided).")
            dataroot_lq = opt.get("dataroot_lq")
            dataroot_gt = opt.get("dataroot_gt")
            if not dataroot_lq or not dataroot_gt:
                msg = "CachedSRDataset requires dataroot_lq and dataroot_gt when cache is absent."
                raise ValueError(msg)
            self._load_from_disk(dataroot_lq, dataroot_gt)
            self.cache_in_ram_active = True

        assert len(self.lq_images) == len(self.gt_images), "Unmatched LQ/GT counts."
        self.length = len(self.lq_images)

    def _load_from_disk(self, dataroot_lq: str, dataroot_gt: str) -> None:
        lq_root = Path(dataroot_lq)
        gt_root = Path(dataroot_gt)
        if not lq_root.exists() or not gt_root.exists():
            msg = f"CachedSRDataset dataroots do not exist. LQ: {dataroot_lq}, GT: {dataroot_gt}"
            raise FileNotFoundError(msg)

        def collect(root: Path) -> dict[str, Path]:
            return {
                str(path.relative_to(root)).replace("\\", "/"): path
                for path in root.rglob("*")
                if path.is_file()
            }

        lq_files = collect(lq_root)
        gt_files = collect(gt_root)
        keys = sorted(set(lq_files) & set(gt_files))
        if not keys:
            msg = "CachedSRDataset could not find any matched LQ/GT pairs."
            raise RuntimeError(msg)
        dropped_lq = len(lq_files) - len(keys)
        dropped_gt = len(gt_files) - len(keys)
        if dropped_lq or dropped_gt:
            self.logger.warning(
                "CachedSRDataset dropping unmatched pairs (LQ: -%d, GT: -%d).",
                dropped_lq,
                dropped_gt,
            )

        self.lq_images = []
        self.gt_images = []
        self.lq_paths = []
        self.gt_paths = []
        self.rel_paths = keys

        for rel in keys:
            lq_img = cv2.imread(str(lq_files[rel]), cv2.IMREAD_UNCHANGED)
            gt_img = cv2.imread(str(gt_files[rel]), cv2.IMREAD_UNCHANGED)
            if lq_img is None or gt_img is None:
                msg = f"Failed to read cached pair {rel}."
                raise RuntimeError(msg)
            lq_img = np.ascontiguousarray(lq_img).astype(np.float32) / 255.0
            gt_img = np.ascontiguousarray(gt_img).astype(np.float32) / 255.0
            self.lq_images.append(lq_img)
            self.gt_images.append(gt_img)
            self.lq_paths.append(str(lq_files[rel]))
            self.gt_paths.append(str(gt_files[rel]))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        lq_img = self.lq_images[index]
        gt_img = self.gt_images[index]
        gt_path = self.gt_paths[index] if index < len(self.gt_paths) else None
        lq_path = self.lq_paths[index] if index < len(self.lq_paths) else None

        # Training branch: random crop + augment
        if self.phase == "train" and self.patch_size:
            gt_img, lq_img = paired_random_crop(
                gt_img, lq_img, self.patch_size, self.scale, gt_path or self.rel_paths[index]
            )
            gt_img = np.ascontiguousarray(gt_img)
            lq_img = np.ascontiguousarray(lq_img)
            gt_img, lq_img = basic_augment(
                [gt_img, lq_img],
                hflip=self.use_hflip,
                rotation=self.use_rot,
            )  # type: ignore[reportAssignmentType]
            gt_img = np.ascontiguousarray(gt_img)
            lq_img = np.ascontiguousarray(lq_img)
        else:
            # Align GT to LQ size when evaluating
            h_lq, w_lq = lq_img.shape[:2]
            gt_img = gt_img[: h_lq * self.scale, : w_lq * self.scale, ...]
            gt_img = np.ascontiguousarray(gt_img)
            lq_img = np.ascontiguousarray(lq_img)

        gt_tensor, lq_tensor = img2tensor(
            [gt_img, lq_img], bgr2rgb=True, float32=True, color=self.color
        )

        if self.mean is not None or self.std is not None:
            normalize(lq_tensor, self.mean, self.std, inplace=True)  # type: ignore[arg-type]
            normalize(gt_tensor, self.mean, self.std, inplace=True)  # type: ignore[arg-type]

        return {
            "lq": lq_tensor,
            "gt": gt_tensor,
            "lq_path": lq_path or self.rel_paths[index],
            "gt_path": gt_path or self.rel_paths[index],
        }
