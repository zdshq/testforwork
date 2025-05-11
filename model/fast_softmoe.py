import torch
from torch import nn


__all__ = ["MultiExpertsMlp"]

class MultiExpertsMlp(nn.Module):
    def __init__(self,
                 in_features:int,
                 hidden_features:int,
                 out_features:int,
                 expert_num:int,
                 act_layer:nn.Module = nn.GELU,
                 ):
        super().__init__()
        # 修改参数名称以符合PyTorch标准命名约定
        # 创建fc1和fc2作为模块容器
        self.fc1 = nn.Module()
        self.fc2 = nn.Module()
        
        # 将权重和偏置作为fc1和fc2的参数
        self.fc1.weight = nn.Parameter(torch.randn(expert_num, in_features, hidden_features))
        self.fc1.bias = nn.Parameter(torch.randn(hidden_features))
        
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p=0.0)
        self.norm = nn.Identity()
        
        self.fc2.weight = nn.Parameter(torch.randn(expert_num, hidden_features, out_features))
        self.fc2.bias = nn.Parameter(torch.randn(out_features))
        self.drop2 = nn.Dropout(p=0.0)

    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # 更新forward函数以使用新的参数名称
        hidden_results = self.norm(self.act(self.drop1(torch.einsum("besd,edh->besh", x, self.fc1.weight) + self.fc1.bias)))
        return self.drop2(torch.einsum("besh,ehd->besd", hidden_results, self.fc2.weight) + self.fc2.bias)