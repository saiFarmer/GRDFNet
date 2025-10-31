import math
from typing import Union
import torch
from typing_extensions import override

from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .arch.GRDFNet import GRDFNet


def _infer_scale(state_dict):
    try:
        expand_weight = state_dict["upsample0.expand.weight"]
        in_ch = expand_weight.shape[1]
        ratio = expand_weight.shape[0] // in_ch
        scale = int(math.isqrt(ratio))
        assert (
            scale * scale == ratio
        ), "Unexpected expand weight shape"  # 1, 2, 4, 8, 16...
        print(f"scale: {scale}")
    except:
        scale = 1

    return scale


def _infer_num_sets(state_dict: dict[str, torch.Tensor]) -> int:
    body_keys = [k for k in state_dict.keys() if k.startswith("body.")]
    block_indices = {k.split(".")[1] for k in body_keys if k.split(".")[1].isdigit()}
    num_blocks = len(block_indices)

    if num_blocks <= 5:
        # minimal model (fallback)
        return 0

    # each set contributes +2 blocks beyond the 5-block stem
    num_sets = max((num_blocks - 5) // 2, 0)
    return num_sets


def compute_receptive_fields(num_sets: int) -> tuple[int, int]:
    # 3xLWGRB + 2xLWDRB + (num_sets x [1 LWGRB + 1 LWDRB])
    n_grb = 3 + num_sets
    n_drb = 2 + num_sets
    trf = 1 + 2 + n_grb * (2 * 1) * 2 + n_drb * (2 * 4) * 2 + 2
    erf = int(trf * 0.7)
    return erf, trf


class GRDFNetArch(Architecture[GRDFNet]):
    def __init__(self) -> None:
        super().__init__(
            id="GRDFNet",
            detect=KeyCondition.has_any(
                "head.weight",
                "tail.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[GRDFNet]:
        num_in_ch: int = 3
        num_out_ch: int = 3
        feature_channels: int = 32
        norm = True
        img_range = 255.0
        rgb_mean = (0.4488, 0.4371, 0.4040)

        if "no_norm" in state_dict:
            norm = False
            state_dict["no_norm"] = torch.zeros(1)

        feature_channels, num_in_ch, _, _ = state_dict["head.weight"].shape
        num_out_ch, _, _, _ = state_dict["tail.weight"].shape

        scale = _infer_scale(state_dict)
        num_sets = _infer_num_sets(state_dict)
        state_params = sum(v.numel() for v in state_dict.values())
        erf, trf = compute_receptive_fields(num_sets)

        model = GRDFNet(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            feature_channels=feature_channels,
            upscale=scale,
            norm=norm,
            img_range=img_range,
            rgb_mean=rgb_mean,
            num_sets=num_sets,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="SR",
            tags=[
                f"{feature_channels}ch",
                f"x{scale}",
                f"nS:{num_sets}",
                f"Params:{state_params}",
                f"ERF:{erf}",
                f"TRF:{trf}",
            ],
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=num_in_ch,
            output_channels=num_out_ch,
        )
