# training/losses.py
"""
Loss functions for video rain removal
Contains: Charbonnier, SSIM, Edge, Temporal, Perceptual, and Combined losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ======================
# 1. Charbonnier Loss (L_pixel)
# ======================
class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss - smooth L1 variant
    sqrt(x^2 + epsilon^2)
    """

    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)


# ======================
# 2. SSIM Loss (L_ssim)
# ======================
class SSIMLoss(nn.Module):
    """
    SSIM Loss - preserves structural similarity
    """

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)

    def _create_window(self, window_size, channel):
        """Create Gaussian window"""

        def gaussian(window_size, sigma=1.5):
            gauss = torch.Tensor([
                torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2.0 * sigma ** 2)))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        window = self.window.to(img1.device).type_as(img1)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, pred, target):
        """
        pred: (B, T, C, H, W) or (B, C, H, W)
        target: same shape
        Returns: 1 - SSIM (so lower is better)
        """
        # Flatten temporal dimension if exists
        if pred.dim() == 5:
            B, T, C, H, W = pred.shape
            pred = pred.view(B * T, C, H, W)
            target = target.view(B * T, C, H, W)

        return 1 - self._ssim(pred, target)


# ======================
# 3. Edge Loss (L_edge)
# ======================
class EdgeLoss(nn.Module):
    """
    Edge Loss - preserves sharp edges
    Compares gradients (Sobel) between output and GT
    """

    def __init__(self):
        super().__init__()
        # Sobel filters
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

    def _get_gradients(self, img):
        """
        Calculate gradients in X and Y directions
        img: (B, C, H, W)
        """
        B, C, H, W = img.shape

        # Expand filters to all channels
        sobel_x = self.sobel_x.repeat(C, 1, 1, 1).to(img.device)
        sobel_y = self.sobel_y.repeat(C, 1, 1, 1).to(img.device)

        grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)

        # Gradient magnitude
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        return grad_magnitude

    def forward(self, pred, target):
        """
        pred: (B, T, C, H, W) or (B, C, H, W)
        target: same shape
        """
        # Flatten temporal dimension
        if pred.dim() == 5:
            B, T, C, H, W = pred.shape
            pred = pred.view(B * T, C, H, W)
            target = target.view(B * T, C, H, W)

        pred_edges = self._get_gradients(pred)
        target_edges = self._get_gradients(target)

        return F.l1_loss(pred_edges, target_edges)


# ======================
# 4. Temporal Consistency Loss (L_temp)
# ======================
class TemporalConsistencyLoss(nn.Module):
    """
    Temporal Consistency Loss - prevents flickering
    Compares frame-to-frame changes
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        pred: (B, T, C, H, W) - T frames
        target: (B, T, C, H, W)
        """
        if pred.dim() != 5:
            # No temporal dimension, return zero loss
            return torch.tensor(0.0, device=pred.device)

        B, T, C, H, W = pred.shape

        if T < 2:
            # Need at least 2 frames
            return torch.tensor(0.0, device=pred.device)

        # Calculate differences between consecutive frames
        pred_diff = pred[:, 1:] - pred[:, :-1]  # (B, T-1, C, H, W)
        target_diff = target[:, 1:] - target[:, :-1]  # (B, T-1, C, H, W)

        # Compare the differences
        return F.l1_loss(pred_diff, target_diff)


# ======================
# 5. Perceptual Loss (L_perc)
# ======================
class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss - optimized version
    Compares high-level features instead of raw pixels
    """

    def __init__(self, layers=[3, 8, 15], weights=[1.0, 0.5, 0.25]):
        """
        layers: which VGG layers to use (default: relu1_2, relu2_2, relu3_3)
        weights: weight for each layer (deeper layers get less weight)
        """
        super().__init__()

        # Load VGG16
        vgg = models.vgg16(pretrained=True).features

        # Split into blocks at specified layers
        self.blocks = nn.ModuleList()
        prev_layer = 0
        for layer_idx in layers:
            block = nn.Sequential(*[vgg[i] for i in range(prev_layer, layer_idx + 1)])
            self.blocks.append(block)
            prev_layer = layer_idx + 1

        # Freeze all parameters
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.weights = weights

        # VGG normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        """Normalize to ImageNet stats"""
        return (x - self.mean) / self.std

    def _extract_features(self, x):
        """Extract features from all blocks"""
        # Flatten temporal dimension if exists
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)

        # Normalize
        x = self._normalize(x)

        # Extract features
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)

        return features

    def forward(self, pred, target):
        """
        pred: (B, T, C, H, W) or (B, C, H, W)
        target: same shape
        """
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)

        loss = 0.0
        for pred_feat, target_feat, weight in zip(pred_features, target_features, self.weights):
            loss += weight * F.l1_loss(pred_feat, target_feat)

        return loss


# ======================
# 6. Combined Loss
# ======================
class CombinedVideoLoss(nn.Module):
    """
    Combined loss for video rain removal

    Loss = α*L_pixel + β*L_ssim + γ*L_edge + δ*L_temp + ε*L_perc

    Args:
        alpha: Charbonnier weight (pixel-level accuracy)
        beta: SSIM weight (structural similarity)
        gamma: Edge weight (sharp edges)
        delta: Temporal weight (temporal consistency)
        epsilon: Perceptual weight (high-level features) - set to 0 to disable
    """

    def __init__(self,
                 alpha=1.0,  # Charbonnier
                 beta=0.3,  # SSIM
                 gamma=0.3,  # Edge
                 delta=0.2,  # Temporal
                 epsilon=0.2):  # Perceptual (0 = disabled)
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.temporal = TemporalConsistencyLoss()

        # Only create perceptual loss if needed
        if epsilon > 0:
            self.perceptual = PerceptualLoss()
        else:
            self.perceptual = None

    def forward(self, pred, target):
        """
        pred: (B, T, C, H, W)
        target: (B, T, C, H, W)

        Returns:
            total_loss: scalar tensor
            loss_dict: dictionary with individual loss components
        """
        zero = pred.new_tensor(0.0)

        loss_pixel = self.charbonnier(pred, target) if self.alpha != 0.0 else zero
        loss_ssim = self.ssim(pred, target) if self.beta != 0.0 else zero
        loss_edge = self.edge(pred, target) if self.gamma != 0.0 else zero
        loss_temp = self.temporal(pred, target) if self.delta != 0.0 else zero

        total_loss = (
                self.alpha * loss_pixel +
                self.beta * loss_ssim +
                self.gamma * loss_edge +
                self.delta * loss_temp
        )

        loss_dict = {
            'pixel': loss_pixel.item(),
            'ssim': loss_ssim.item(),
            'edge': loss_edge.item(),
            'temporal': loss_temp.item(),
        }

        # Add perceptual loss if enabled
        if self.perceptual is not None and self.epsilon != 0.0:
            loss_perc = self.perceptual(pred, target)
            total_loss += self.epsilon * loss_perc
            loss_dict['perceptual'] = loss_perc.item()
        else:
            loss_dict['perceptual'] = 0.0

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict