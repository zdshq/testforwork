import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional

class MultiExpertsMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        expert_num: int,
        act_layer: Type[nn.Module] = nn.GELU,
        drop1: float = 0.1,
        drop2: float = 0.1,
        norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop1)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.drop2 = nn.Dropout(drop2)

        # 保持原有命名：fc1、fc2
        self.fc1 = nn.Module()
        self.fc2 = nn.Module()
        # 权重与偏置
        self.fc1.weight = nn.Parameter(torch.randn(expert_num, in_features, hidden_features))
        self.fc1.bias   = nn.Parameter(torch.randn(hidden_features))
        self.fc2.weight = nn.Parameter(torch.randn(expert_num, hidden_features, out_features))
        self.fc2.bias   = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一阶段
        h = torch.einsum("besd,edh->besh", x, self.fc1.weight) + self.fc1.bias
        h = self.act(h)
        h = self.drop1(h)
        h = self.norm(h)
        # 第二阶段
        out = torch.einsum("besh,ehd->besd", h, self.fc2.weight) + self.fc2.bias
        out = self.drop2(out)
        # 残差
        if self.residual:
            out = out + x
        return out

