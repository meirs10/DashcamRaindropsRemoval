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
        # in_channels will be inferred on first forward from [x, h]
        self.conv = nn.LazyConv2d(4 * hidden_dim, kernel_size=kernel_size,
                                  padding=padding, bias=bias)

    def forward(self, x, state=None):
        """
        x:      (B, C_in, H, W)
        state:  None OR (h, c) with h,c: (B, hidden_dim, H, W)
        """
        if state is None or state[0] is None:  # Handle None or (None, None)
            B, _, H, W = x.shape
            device = x.device
            h = torch.zeros(B, self.hidden_dim, H, W, device=device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=device, dtype=x.dtype)
        else:
            h, c = state

        # concat along channels
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
    We automatically pick 4 resolution stages based on where spatial size changes.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = mobilenet_v3_small(pretrained=pretrained)
        self.features = backbone.features  # nn.Sequential

        # Find indices where spatial size changes and keep last 4 stages
        self.stage_indices = self._compute_stage_indices()

    def _compute_stage_indices(self):
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

        # keep 4 deepest scales
        if len(indices) >= 4:
            indices = indices[-4:]
        return indices

    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns list [f1, f2, f3, f4] with descending resolutions
        (f1 highest resolution, f4 deepest/bottleneck).
        """
        feats = []
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self.stage_indices:
                feats.append(h)

        # safety check
        assert len(feats) == 4, f"Expected 4 feature maps, got {len(feats)}"
        return feats  # [f1, f2, f3, f4]


# ======================
# U-Net style decoder
# ======================
class UpBlock(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        # in_channels for convs will be inferred (skip + upsampled)
        self.conv1 = nn.LazyConv2d(out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.LazyConv2d(out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # Upsample and concat with skip
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class UNetDecoder(nn.Module):
    def __init__(self, out_channels: int = 3):
        super().__init__()
        # Decoder channels chosen small for speed; LazyConv2d figures the inputs.
        self.up3 = UpBlock(128)  # bottleneck + f3
        self.up2 = UpBlock(64)  # up3 + f2
        self.up1 = UpBlock(32)  # up2 + f1

        self.final_conv = nn.Sequential(
            nn.LazyConv2d(16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(out_channels, kernel_size=1)
        )

    def forward(self, bottleneck, enc_feats, out_size):
        """
        bottleneck: (B, Cb, Hb, Wb)       (from ConvLSTM)
        enc_feats:  [f1, f2, f3, f4]     (from encoder)
        out_size:   (H, W) original frame size
        """
        f1, f2, f3, f4 = enc_feats  # f4 is the deepest encoder feature (same spatial size as bottleneck)

        x = self.up3(bottleneck, f3)  # -> ~f3 size
        x = self.up2(x, f2)  # -> ~f2 size
        x = self.up1(x, f1)  # -> ~f1 size

        x = self.final_conv(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


# ======================
# Full Video De-Raindrop Network
# ======================
class MobileNetV3UNetConvLSTMVideo(nn.Module):
    """
    Video deraining model:
    - Encoder: MobileNetV3-Small (pretrained) → 4 scales
    - Bottleneck: 1 Conv layer + ConvLSTM over time
    - Decoder: U-Net style, shared across frames
    Input:  (B, T, 3, H, W)
    Output: (B, T, 3, H, W)
    """

    def __init__(self,
                 hidden_dim: int = 96,
                 out_channels: int = 3,
                 use_pretrained_encoder: bool = True,
                 freeze_encoder: bool = True):
        super().__init__()
        self.encoder = MobileNetV3Encoder(pretrained=use_pretrained_encoder)

        # Freeze encoder if requested
        if freeze_encoder:
            self.freeze_encoder()

        # Project encoder bottleneck (f4) to ConvLSTM hidden size
        self.bottleneck_proj = nn.LazyConv2d(hidden_dim, kernel_size=1)

        self.convlstm = ConvLSTMCell(hidden_dim=hidden_dim, kernel_size=3)
        self.decoder = UNetDecoder(out_channels=out_channels)

    def freeze_encoder(self):
        """Freeze all encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen (ImageNet weights preserved)")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("✓ Encoder unfrozen")

    def get_trainable_params(self):
        """Get count of trainable vs frozen parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable
        return {
            'trainable': trainable,
            'frozen': frozen,
            'total': total,
            'trainable_pct': 100 * trainable / total if total > 0 else 0
        }

    def print_param_summary(self):
        """Print parameter summary"""
        stats = self.get_trainable_params()
        print("\n" + "=" * 60)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 60)
        print(f"Trainable parameters: {stats['trainable']:,}")
        print(f"Frozen parameters:    {stats['frozen']:,}")
        print(f"Total parameters:     {stats['total']:,}")
        print(f"Trainable %:          {stats['trainable_pct']:.1f}%")
        print("=" * 60 + "\n")

    def forward(self, x):
        """
        x: (B, T, 3, H, W)
        returns: (B, T, 3, H, W)
        """
        B, T, C, H, W = x.shape
        outputs = []
        h_state, c_state = None, None

        for t in range(T):
            frame = x[:, t]  # (B, 3, H, W)

            # Encoder features (frozen, no gradients computed)
            with torch.set_grad_enabled(
                    self.encoder.training and any(p.requires_grad for p in self.encoder.parameters())):
                enc_feats = self.encoder(frame)  # [f1, f2, f3, f4]

            f1, f2, f3, f4 = enc_feats

            # Bottleneck + ConvLSTM across time (trainable)
            bottleneck = self.bottleneck_proj(f4)

            # Pass state properly (None on first frame, tuple afterwards)
            if h_state is None:
                h_state, c_state = self.convlstm(bottleneck, None)
            else:
                h_state, c_state = self.convlstm(bottleneck, (h_state, c_state))

            # Decode current frame using temporally-enhanced bottleneck + current skips (trainable)
            out_frame = self.decoder(h_state, enc_feats, out_size=(H, W))  # (B, 3, H, W)
            outputs.append(out_frame.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (B, T, 3, H, W)


# ======================
# Usage Example
# ======================
if __name__ == "__main__":
    # Create model with frozen encoder
    model = MobileNetV3UNetConvLSTMVideo(
        hidden_dim=96,
        out_channels=3,
        use_pretrained_encoder=True,
        freeze_encoder=True  # Freeze encoder for transfer learning
    )

    # Print parameter summary
    model.print_param_summary()

    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dummy_input = torch.randn(1, 5, 3, 256, 256).to(device)
    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print("\n✓ Model ready for training!")