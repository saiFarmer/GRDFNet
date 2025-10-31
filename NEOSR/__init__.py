import importlib
import os
import random
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data
from torch.utils import data
from torch.utils.data.sampler import Sampler

from neosr.utils import get_root_logger, scandir
from neosr.utils.dist_util import get_dist_info
from neosr.utils.registry import DATASET_REGISTRY

__all__ = ["build_dataloader", "build_dataset"]

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = Path(Path(__file__).resolve()).parent
dataset_filenames = [
    Path(Path(v).name).stem
    for v in scandir(str(data_folder))
    if v.endswith("_dataset.py")
]

def build_dataset(dataset_opt: dict[str, Any]):
    """Build dataset from options.

    Args:
    ----
        dataset_opt (dict): Configuration for dataset. It must contain:
            type (str): Dataset type.

    """
    dataset_opt = deepcopy(dataset_opt)
    dataset_type = dataset_opt["type"]
    dataset_cls = DATASET_REGISTRY.get(dataset_type)  # type: ignore[assignment]
    logger = get_root_logger()

    def _maybe_cache_to_ram() -> None:
        default_cache = str(dataset_type).lower().startswith("cached")
        cache_requested = dataset_opt.get("cache_in_ram", default_cache)
        dataroot_lq = dataset_opt.get("dataroot_lq")
        dataroot_gt = dataset_opt.get("dataroot_gt")
        if not cache_requested:
            logger.debug(
                "Skipping RAM cache preload for dataset '%s' (cache_in_ram disabled).",
                dataset_type,
            )
            return
        if not dataroot_lq or not dataroot_gt:
            logger.warning(
                "cache_in_ram enabled for dataset '%s' but dataroots are missing. Skipping preload.",
                dataset_type,
            )
            return

        lq_root = Path(dataroot_lq)
        gt_root = Path(dataroot_gt)
        if not lq_root.exists() or not gt_root.exists():
            logger.warning(
                "cache_in_ram enabled for dataset '%s' but dataroots do not exist. "
                "LQ: %s, GT: %s. Skipping preload.",
                dataset_type,
                dataroot_lq,
                dataroot_gt,
            )
            return

        try:
            from cv2 import IMREAD_UNCHANGED, imread  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover - OpenCV should be available
            logger.warning(
                "cache_in_ram enabled for dataset '%s' but OpenCV is unavailable: %s",
                dataset_type,
                exc,
            )
            return

        def collect_files(root: Path) -> dict[str, Path]:
            return {
                str(path.relative_to(root)).replace("\\", "/"): path
                for path in root.rglob("*")
                if path.is_file()
            }

        lq_files = collect_files(lq_root)
        gt_files = collect_files(gt_root)
        common_rel_paths = sorted(set(lq_files) & set(gt_files))
        if not common_rel_paths:
            logger.warning(
                "cache_in_ram enabled for dataset '%s' but no matching LQ/GT pairs were found.",
                dataset_type,
            )
            return

        if len(common_rel_paths) != len(lq_files) or len(common_rel_paths) != len(gt_files):
            logger.info(
                "cache_in_ram: restricting to %d matched pairs (dropped %d LQ, %d GT).",
                len(common_rel_paths),
                len(lq_files) - len(common_rel_paths),
                len(gt_files) - len(common_rel_paths),
            )

        lq_images: list[np.ndarray] = []
        gt_images: list[np.ndarray] = []
        total_bytes = 0
        for rel_path in common_rel_paths:
            lq_img = imread(str(lq_files[rel_path]), IMREAD_UNCHANGED)
            gt_img = imread(str(gt_files[rel_path]), IMREAD_UNCHANGED)
            if lq_img is None or gt_img is None:
                logger.warning(
                    "cache_in_ram: failed to read pair '%s'. Aborting preload.",
                    rel_path,
                )
                lq_images.clear()
                gt_images.clear()
                break
            lq_img = np.ascontiguousarray(lq_img).astype(np.float32) / 255.0
            gt_img = np.ascontiguousarray(gt_img).astype(np.float32) / 255.0
            lq_images.append(lq_img)
            gt_images.append(gt_img)
            total_bytes += lq_img.nbytes + gt_img.nbytes
        if not lq_images or not gt_images:
            return

        dataset_opt["_cached_lq_images"] = lq_images
        dataset_opt["_cached_gt_images"] = gt_images
        dataset_opt["_cached_rel_paths"] = common_rel_paths
        dataset_opt["_cached_lq_paths"] = [str(lq_files[p]) for p in common_rel_paths]
        dataset_opt["_cached_gt_paths"] = [str(gt_files[p]) for p in common_rel_paths]
        dataset_opt.setdefault("scale", dataset_opt.get("scale", 1))
        logger.info(
            "Preloaded %d image pairs for dataset '%s' into RAM (approx %.2f GiB).",
            len(common_rel_paths),
            dataset_type,
            total_bytes / (1024**3),
        )

    _maybe_cache_to_ram()
    dataset = dataset_cls(dataset_opt)  # type: ignore[operator]
    cache_active = getattr(dataset, "cache_in_ram_active", False) or bool(
        dataset_opt.get("_cached_lq_images")
    )
    if cache_active:
        setattr(dataset, "cache_in_ram_active", True)
    logger.info(f"Dataset [{dataset.__class__.__name__}] is built.")
    return dataset


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int) -> None:
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    # NOTE: set seed on old generator as a precaution, but
    # it is redundand since we use np.random.Generator
    np.random.seed(worker_seed)  # noqa: NPY002
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    dataset: data.Dataset,
    dataset_opt: dict[str, Any],
    num_gpu: int = 1,
    dist: bool = False,
    sampler: Sampler | None = None,
    seed: int | None = None,
) -> data.DataLoader:
    """Build dataloader.

    Args:
    ----
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None

    """
    phase = dataset_opt["phase"]
    rank, _ = get_dist_info()
    logger = get_root_logger()

    # train
    if phase == "train":
        cpu_count = os.cpu_count() or 1
        workers_per_gpu_opt = dataset_opt.get("num_worker_per_gpu")
        # Heuristic tuned for high-end hosts: keep workers high enough to saturate I/O.
        if workers_per_gpu_opt is None or workers_per_gpu_opt == "auto":
            gpu_slots = num_gpu if num_gpu > 0 else 1
            workers_per_gpu = max(4, min(32, cpu_count // gpu_slots))
        else:
            workers_per_gpu = int(workers_per_gpu_opt)
        workers_per_gpu = max(workers_per_gpu, 1)
        num_workers = workers_per_gpu

        if dist:  # distributed training
            batch_size = dataset_opt["batch_size"]
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt["batch_size"] * multiplier
            num_workers *= multiplier

        if "prefetch_factor" in dataset_opt:
            prefetch_factor = dataset_opt["prefetch_factor"]
        else:
            # High worker counts benefit from deeper prefetch queues.
            prefetch_factor = max(4, min(16, workers_per_gpu))

        cache_in_ram_active = getattr(dataset, "cache_in_ram_active", False)
        if os.name == "nt" and cache_in_ram_active and num_workers > 0:
            logger.warning(
                "Windows detected with RAM-cached dataset '%s'; forcing num_workers from %d to 0 "
                "to avoid spawn pickling overhead.",
                dataset.__class__.__name__,
                num_workers,
            )
            num_workers = 0

        dataloader_args = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "sampler": sampler,
            "drop_last": True,
        }
        if num_workers > 0:
            dataloader_args["prefetch_factor"] = prefetch_factor
        if sampler is None:
            dataloader_args["shuffle"] = True
        dataloader_args["worker_init_fn"] = (
            partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
            if seed is not None
            else None
        )

    # val
    elif phase in {"val", "test"}:
        dataloader_args = {
            "dataset": dataset,
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
        }
    else:
        msg = f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'."
        raise ValueError(msg)

    dataloader_args["pin_memory"] = dataset_opt.get("pin_memory", True)
    if "persistent_workers" in dataset_opt:
        dataloader_args["persistent_workers"] = dataset_opt["persistent_workers"]
    else:
        dataloader_args["persistent_workers"] = dataloader_args.get("num_workers", 0) > 0
        if dataloader_args["persistent_workers"] and dataloader_args["num_workers"] == 0:
            dataloader_args["persistent_workers"] = False

    return data.DataLoader(**dataloader_args)
