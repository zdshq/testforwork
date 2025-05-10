
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
        #   (mlp): Mlp(
        #     (fc1): Linear(in_features=192, out_features=768, bias=True)
        #     (act): GELU(approximate='none')
        #     (drop1): Dropout(p=0.0, inplace=False)
        #     (norm): Identity()
        #     (fc2): Linear(in_features=768, out_features=192, bias=True)
        #     (drop2): Dropout(p=0.0, inplace=False)
    #   )
        self.fc1_weights = nn.Parameter(torch.randn(expert_num, in_features, hidden_features))
        self.fc1_biases = nn.Parameter(torch.randn(expert_num, hidden_features))
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p=0.0)
        self.norm = nn.Identity()
        self.fc2_weights = nn.Parameter(torch.randn(expert_num, hidden_features, out_features))
        self.fc2_biases = nn.Parameter(torch.randn(expert_num, out_features))
        self.drop2 = nn.Dropout(p=0.0)

    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        hidden_results = self.norm(self.act(self.drop1(torch.einsum("besd,edh->besh", x, self.fc1_weights) + self.fc1_biases)))
        return self.drop2(torch.einsum("besh,ehd->besd", hidden_results, self.fc2_weights) + self.fc2_biases)

        