"""依据公开行为契约实现的小型神经排序结构。

输入统一使用 ``[batch, stocks, time, features]``。跨股票模块不加入股票顺序
位置编码，因此置换横截面只会同步置换输出分数。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_panel(values: torch.Tensor, feature_dim: int) -> tuple[int, int, int]:
    if values.ndim != 4:
        raise ValueError('neural ranker input must have shape [batch, stocks, time, features]')
    batch, stocks, steps, features = values.shape
    if min(batch, stocks, steps) < 1 or features != feature_dim:
        raise ValueError(f'expected feature dimension {feature_dim}, received {features}')
    return batch, stocks, steps


def _temporal_positions(steps: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(steps, device=device, dtype=dtype).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10_000.0) / width)
    )
    encoding = torch.zeros((steps, width), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


class CrossStockHead(nn.Module):
    """先执行置换等变 attention，再为每只股票独立评分。"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        context, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        return self.scorer(self.normalization(encoded + context)).squeeze(-1)


class TemporalTransformerRanker(nn.Module):
    """E0：单股时序 Transformer 加跨股票 attention。"""

    def __init__(self, feature_dim: int, hidden_dim: int = 32, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.cross_stock = CrossStockHead(hidden_dim, num_heads, dropout)

    def encode_sequence(self, values: torch.Tensor) -> torch.Tensor:
        batch, stocks, steps = _validate_panel(values, self.feature_dim)
        sequence = self.input_projection(values).reshape(batch * stocks, steps, -1)
        sequence = sequence + _temporal_positions(steps, sequence.shape[-1], sequence.device, sequence.dtype)
        encoded = self.temporal_encoder(sequence)
        return encoded.reshape(batch, stocks, steps, -1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.cross_stock(self.encode_sequence(values)[:, :, -1])


class CausalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.convolution = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        convolved = self.convolution(F.pad(values, (self.left_padding, 0)))
        residual = values + self.dropout(F.gelu(convolved))
        return self.normalization(residual.transpose(1, 2)).transpose(1, 2)


class CausalTCNRanker(nn.Module):
    """E2：低成本因果时序卷积排序器。"""

    def __init__(self, feature_dim: int, hidden_dim: int = 32, dropout: float = 0.1, **_: object) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.input_projection = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            CausalResidualBlock(hidden_dim, dilation=1, dropout=dropout),
            CausalResidualBlock(hidden_dim, dilation=2, dropout=dropout),
        )
        self.scorer = nn.Linear(hidden_dim, 1)

    def encode_sequence(self, values: torch.Tensor) -> torch.Tensor:
        batch, stocks, steps = _validate_panel(values, self.feature_dim)
        sequence = values.reshape(batch * stocks, steps, self.feature_dim).transpose(1, 2)
        encoded = self.blocks(self.input_projection(sequence)).transpose(1, 2)
        return encoded.reshape(batch, stocks, steps, -1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.scorer(self.encode_sequence(values)[:, :, -1]).squeeze(-1)


class TemporalCrossStockAttentionRanker(nn.Module):
    """E3：单股循环时序编码加跨股票 attention。"""

    def __init__(self, feature_dim: int, hidden_dim: int = 32, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.temporal_encoder = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.cross_stock = CrossStockHead(hidden_dim, num_heads, dropout)

    def encode_sequence(self, values: torch.Tensor) -> torch.Tensor:
        batch, stocks, steps = _validate_panel(values, self.feature_dim)
        sequence = values.reshape(batch * stocks, steps, self.feature_dim)
        encoded, _ = self.temporal_encoder(sequence)
        return encoded.reshape(batch, stocks, steps, -1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.cross_stock(self.encode_sequence(values)[:, :, -1])


def build_ranker(
    family: str,
    feature_dim: int,
    hidden_dim: int = 32,
    num_heads: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    """构造一个已声明的实验模型族，不修改生产配置。"""
    constructors = {
        'e0_transformer': TemporalTransformerRanker,
        'e2_tcn': CausalTCNRanker,
        'e3_cross_stock': TemporalCrossStockAttentionRanker,
    }
    if family not in constructors:
        raise ValueError(f'unsupported neural structure family: {family}')
    return constructors[family](
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    )
