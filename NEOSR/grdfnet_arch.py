import torch
from torch import nn

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


def conv3x3(in_channels, out_channels, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
    )


class LWGRB(nn.Module):
    def __init__(self, channels: int, bias: bool = True, identity=True):
        super().__init__()
        self.identity = identity
        self.conv1 = conv3x3(channels, channels, bias)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv3x3(channels, channels, bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.conv2(self.act(self.conv1(x)))
        a = torch.sigmoid(r)
        return x + r * a


class LWDRB(nn.Module):
    def __init__(self, c, bias=True, dil=4):
        super().__init__()
        self.conv1 = conv3x3(c, c, bias)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(c, c, 3, padding=dil, dilation=dil, bias=bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        r = self.conv2(self.act(self.conv1(x)))
        return x + r


class LWGRBShuffle(nn.Module):
    def __init__(self, in_ch, out_ch, scale, bias: bool = True):
        super().__init__()
        self.expand = conv3x3(in_ch, out_ch * scale * scale, bias)
        self.up = nn.PixelShuffle(scale)
        self.refine = LWGRB(out_ch, bias=bias)

    def forward(self, x):
        x = self.expand(x)
        x = self.up(x)
        return self.refine(x)


@ARCH_REGISTRY.register()
class GRDFNet(nn.Module):
    """
    GRDFNet (Gated + Residual Dilated Fast Network)
    Configurable block stacking via integer `num_sets`.

    Structure:
      Stem: 3xLWGRB + 2xLWDRB
      Repeated: num_sets x [LWGRB + LWDRB]
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        feature_channels: int = 32,
        upscale: int = upscale,
        bias: bool = True,
        norm: bool = False,
        img_range: float = 1.0,
        rgb_mean=(0.5, 0.5, 0.5),
        num_sets: int = 3,
    ):
        super().__init__()

        self.in_ch = num_in_ch
        self.out_ch = num_out_ch
        self.c = feature_channels
        self.scale = upscale
        self.img_range = img_range
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.num_sets = num_sets

        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        if not norm:
            self.register_buffer("no_norm", torch.zeros(1))
        else:
            self.no_norm = None

        self.head = conv3x3(self.in_ch, self.c, bias)
        self.body = self._make_body(bias)
        self.tail = conv3x3(self.c, self.out_ch, bias)

        if self.scale == 1:
            self.upsample0 = nn.Identity()
        else:
            self.upsample0 = LWGRBShuffle(
                self.out_ch, self.out_ch, self.scale, bias=bias
            )

    def _make_body(self, bias: bool):
        blocks = [
            LWGRB(self.c, bias),
            LWGRB(self.c, bias),
            LWGRB(self.c, bias),
            LWDRB(self.c, bias),
            LWDRB(self.c, bias),
        ]
        for _ in range(self.num_sets):
            blocks += [LWGRB(self.c, bias), LWDRB(self.c, bias)]
        return nn.Sequential(*blocks)

    @property
    def is_norm(self) -> bool:
        return getattr(self, "no_norm", None) is None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_norm:
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range

        feat = self.head(x)
        feat = self.body(feat)
        out_feat = self.tail(feat)
        out_feat = (1.0 - self.gamma) * x + self.gamma * out_feat

        out = self.upsample0(out_feat)

        if self.is_norm:
            out = out / self.img_range + self.mean

        return out
