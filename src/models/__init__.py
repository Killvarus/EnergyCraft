"""
Архитектуры моделей FWI.

- FWIMLP: полносвязный baseline
- FWI_PINN: физически-информированная нейросеть
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.config import (
    HEIGHT, WIDTH,
    MLP_HIDDEN_DIMS, MLP_ACTIVATION, MLP_USE_LAYERNORM, MLP_DROPOUT,
)


# ---------------------------------------------------------------------------
# Фабрика активаций
# ---------------------------------------------------------------------------
def _get_activation(name: str) -> nn.Module:
    mapping = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "sin": _Sine(),
    }
    if name not in mapping:
        raise ValueError(f"Неизвестная активация: {name}. Доступны: {list(mapping.keys())}")
    return mapping[name]


class _Sine(nn.Module):
    """Активация sin, как в SIREN (для PINN)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


# ============================================================================
# FWIMLP
# ============================================================================
class FWIMLP(nn.Module):
    """
    Полносвязная FWI-модель:
        [B, 1, H, W] → плоский вектор → MLP → [B, 1, H, W].

    Настройка через config.py:
        MLP_HIDDEN_DIMS   — кортеж размеров скрытых слоёв
        MLP_ACTIVATION    — функция активации
        MLP_USE_LAYERNORM — LayerNorm между слоями
        MLP_DROPOUT       — dropout (0.0 = без)
    """

    def __init__(
        self,
        height: int = HEIGHT,
        width: int = WIDTH,
        hidden_dims: Tuple[int, ...] = MLP_HIDDEN_DIMS,
        activation: str = MLP_ACTIVATION,
        use_layernorm: bool = MLP_USE_LAYERNORM,
        dropout: float = MLP_DROPOUT,
    ) -> None:
        super().__init__()
        input_dim = height * width

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(in_dim, hdim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hdim))
            layers.append(_get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hdim
        layers.append(nn.Linear(in_dim, input_dim))

        self.mlp = nn.Sequential(*layers)
        self.output_shape = (height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.reshape(B, -1)
        out = self.mlp(x)
        return out.reshape(B, 1, *self.output_shape)


# ============================================================================
# FWICNN — сверточный encoder-decoder baseline
# ============================================================================
class FWICNN(nn.Module):
    """
    Компактный сверточный encoder-decoder для отображения
    [B, 1, H, W] (окно сейсмограммы) -> [B, 1, H, W] (окно Vp).

    В отличие от flatten-MLP использует локальную структуру изображения,
    имеет ~1-2M параметров и значительно лучше обобщается на малых выборках.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        ch = base_channels

        def conv_block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = conv_block(1, ch)
        self.enc2 = conv_block(ch, ch * 2)
        self.enc3 = conv_block(ch * 2, ch * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(ch * 4, ch * 4)

        # Decoder (с skip-connections)
        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(ch * 8, ch * 2)
        self.up2 = nn.ConvTranspose2d(ch * 2, ch * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(ch * 4, ch)
        self.up1 = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)
        self.dec1 = conv_block(ch * 2, ch)

        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Подгоняет spatial-размер x к ref (нечетные размеры после pooling)."""
        if x.shape[-2:] != ref.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size = x.shape[-2:]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self._match(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))

        out = self.head(self._match(d1, x))
        if out.shape[-2:] != in_size:
            out = torch.nn.functional.interpolate(out, size=in_size, mode="bilinear", align_corners=False)
        return out


from src.models.pinn_fwi import FWI_PINN

__all__ = ["FWIMLP", "FWICNN", "FWI_PINN"]
