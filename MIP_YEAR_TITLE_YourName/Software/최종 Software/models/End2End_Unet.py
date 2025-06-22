import torch
import torch.nn as nn
import torch.nn.functional as F
from .Unet import UNET

# from ResUnet import ResUnet
# from Unet_Attention import AttU_Net

class SpectralReducer(nn.Module):
    def __init__(self,
                 in_channels: int = 214,
                 hidden_channels: int = 128,
                 out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            # 1×1 Conv == 픽셀별 fully-connected
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 214, H, W) → (B, 3, H, W)
        return self.net(x)


class End2EndUNet(nn.Module):
    """
    Spectral Reducer + Unet -> End-to-End
    """
    def __init__(self,
                 spec_in: int = 214,
                 spec_hidden: int = 128,
                 spec_out: int = 3,
                 num_classes: int = 4,
                 unet_features: list = [64, 128, 256, 512]
                ):
        super().__init__()
        self.reducer = SpectralReducer(
            in_channels=spec_in,
            hidden_channels=spec_hidden,
            out_channels=spec_out
        )
        # UNET(in_channels, out_channels, features)
        self.unet = UNET(
            in_channels=spec_out,
            out_channels=num_classes,
            features=unet_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_reduced = self.reducer(x)
        out = self.unet(x_reduced)
        return out
