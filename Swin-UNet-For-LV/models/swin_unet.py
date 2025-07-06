# models/swin_unet.py
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUnetLV(nn.Module):
    """
    Swin-UNETR wrapper for LV segmentation.
    Input  : (B, T, C, H, W)  --> permute to (B, C, T, H, W)
    Output : (B, T, 1, H, W)
    """
    def __init__(self, img_size: int = 224, pretrained: bool = True):
        super().__init__()
        # SwinUNETR will infer spatial dims; img_size kept for backward-compat
        self.net = SwinUNETR(
            img_size=img_size,
            in_channels=3,
            out_channels=1,
            feature_size=48,
            use_checkpoint=True,
        )
        if pretrained:
            try:
                from monai.networks.nets.swin_unetr import get_swin_unetr_img_size
                _ = get_swin_unetr_img_size(img_size)
            except Exception:
                print("[INFO] Pre-trained weights not found; training from scratch.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape : (B, T, C, H, W)  →  net expects (B, C, D, H, W)
        """
        x = x.permute(0, 2, 1, 3, 4)        # (B, C, T, H, W)
        out = self.net(x)                   # (B, 1, T, H, W)
        out = out.permute(0, 2, 1, 3, 4)    # back to (B, T, 1, H, W)
        return out
