# training/model.py
"""
Model definition for video rain removal.

- MobileNetV3 encoder (pretrained on ImageNet).
- U-Net style decoder.
- ConvLSTM bottleneck for temporal modeling.
- Encoder can be fully frozen, including BatchNorm running stats.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


# ======================
# ConvLSTM cell
# ======================
class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell with LazyConv2d so we don't need to know input channels a priori.
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.LazyConv2d(
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, state=None):
        """
        x:      (B, C_in, H, W)
        state:  None OR (h, c) with h,c: (B, hidden_dim, H, W)
        """
        if state is None or state[0] is None:
            B, _, H, W = x.shape
            device = x.device
            h = torch.zeros(B, self.hidden_dim, H, W, device=device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        i, f, o, g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# ======================
# MobileNetV3 encoder
# ======================
class MobileNetV3Encoder(nn.Module):
    """
    Wrapper around torchvision mobilenet_v3_small to produce multi-scale feature maps.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = mobilenet_v3_small(pretrained=pretrained)
        self.features = backbone.features
        self.stage_indices = self._compute_stage_indices()

    def _compute_stage_indices(self):
        """
        Run a dummy tensor through the feature extractor and record
        indices where spatial resolution changes. Keep last 4.
        """
        indices = []
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            h_prev, w_prev = x.shape[-2:]
            for i, layer in enumerate(self.features):
                x = layer(x)
                h, w = x.shape[-2:]
                if h != h_prev or w != w_prev:
                    indices.append(i)
                    h_prev, w_prev = h, w
        if len(indices) >= 4:
            indices = indices[-4:]
        return indices

    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns: list of 4 feature maps from shallow -> deep.
        """
        feats = []
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self.stage_indices:
                feats.append(h)
        assert len(feats) == 4, f"Expected 4 feature maps, got {len(feats)}"
        return feats


# ======================
# U-Net decoder blocks
# ======================
class UpBlock(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.conv1 = nn.LazyConv2d(out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.LazyConv2d(out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # upsample bottleneck to match skip size
        x = torch.nn.functional.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class UNetDecoder(nn.Module):
    def __init__(self, out_channels: int = 3):
        super().__init__()
        self.up3 = UpBlock(128)
        self.up2 = UpBlock(64)
        self.up1 = UpBlock(32)
        self.final_conv = nn.Sequential(
            nn.LazyConv2d(16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(out_channels, kernel_size=1),
        )

    def forward(self, bottleneck, enc_feats, out_size):
        f1, f2, f3, f4 = enc_feats
        x = self.up3(bottleneck, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        x = self.final_conv(x)
        x = torch.nn.functional.interpolate(
            x, size=out_size, mode="bilinear", align_corners=False
        )
        return x


# ======================
# Helpers for freezing encoder
# ======================
def _freeze_encoder_module(encoder: nn.Module):
    """
    Freeze all encoder parameters and BatchNorm running stats.
    """
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    for m in encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def _unfreeze_encoder_module(encoder: nn.Module):
    """
    Unfreeze encoder parameters (BatchNorm state will then follow train/eval).
    """
    for p in encoder.parameters():
        p.requires_grad = True
    # encoder.train()/eval() will be controlled by outer model.train()


# ======================
# Full model with ConvLSTM
# ======================
class MobileNetV3UNetConvLSTMVideo(nn.Module):
    """
    Encoder–ConvLSTM–Decoder video model.

    Input:  x of shape (B, T, 3, H, W)
    Output: y of shape (B, T, 3, H, W)
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        out_channels: int = 3,
        use_pretrained_encoder: bool = True,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = MobileNetV3Encoder(pretrained=use_pretrained_encoder)

        self.encoder_frozen = False
        if freeze_encoder:
            self.freeze_encoder()

        self.bottleneck_proj = nn.LazyConv2d(hidden_dim, kernel_size=1)
        self.convlstm = ConvLSTMCell(hidden_dim=hidden_dim, kernel_size=3)
        self.decoder = UNetDecoder(out_channels=out_channels)

    # ----- Freezing / unfreezing -----
    def freeze_encoder(self):
        _freeze_encoder_module(self.encoder)
        self.encoder_frozen = True
        print("✓ Encoder frozen (weights + BatchNorm running stats)")

    def unfreeze_encoder(self):
        _unfreeze_encoder_module(self.encoder)
        self.encoder_frozen = False
        print("✓ Encoder unfrozen")

    def train(self, mode: bool = True):
        """
        Override nn.Module.train so that when encoder_frozen is True,
        the encoder (and its BatchNorm) always stays in eval mode.
        """
        super().train(mode)
        if self.encoder_frozen:
            _freeze_encoder_module(self.encoder)
        return self

    # ----- Utility: parameter stats -----
    def get_trainable_params(self) -> Dict[str, float]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total,
            "trainable_pct": 100.0 * trainable / total if total > 0 else 0.0,
        }

    def print_param_summary(self):
        stats = self.get_trainable_params()
        print("\n" + "=" * 60)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 60)
        print(f"Trainable parameters: {stats['trainable']:,}")
        print(f"Frozen parameters:    {stats['frozen']:,}")
        print(f"Total parameters:     {stats['total']:,}")
        print(f"Trainable %:          {stats['trainable_pct']:.1f}%")
        print("=" * 60 + "\n")

    # ----- Forward -----
    def forward(
            self,
            x: torch.Tensor,
            state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            return_state: bool = True,
    ):
        """
        x: (B, T, C, H, W)
        state: (h_state, c_state) where each is (B, hidden_dim, H', W') matching convlstm spatial size.
        return_state: if True returns (y, new_state), else returns y only.
        """
        B, T, C, H, W = x.shape
        outputs = []

        encoder_has_trainable = any(p.requires_grad for p in self.encoder.parameters())

        # Use provided state (for realtime persistence)
        if state is None:
            h_state, c_state = None, None
        else:
            h_state, c_state = state

        for t in range(T):
            frame = x[:, t]  # (B, C, H, W)

            # Encoder: if frozen, no grad and BN stats fixed.
            with torch.set_grad_enabled(self.encoder.training and encoder_has_trainable):
                enc_feats = self.encoder(frame)

            f1, f2, f3, f4 = enc_feats
            bottleneck = self.bottleneck_proj(f4)

            if h_state is None:
                h_state, c_state = self.convlstm(bottleneck, None)
            else:
                h_state, c_state = self.convlstm(bottleneck, (h_state, c_state))

            out_frame = self.decoder(h_state, enc_feats, out_size=(H, W))
            outputs.append(out_frame.unsqueeze(1))

        y = torch.cat(outputs, dim=1)  # (B, T, 3, H, W)
        new_state = (h_state, c_state)

        if return_state:
            return y, new_state
        return y