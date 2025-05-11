import torch
import timm
from model.vision_transformer_zeke import VisionTransformer_zeke

# 1. 创建 ViT-Tiny 模型，并加载 ImageNet 预训练权重
dense_model = timm.create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=False, checkpoint_path="./vit_tiny_patch16_224.augreg_in21k.pth") 

# (可选) 将模型移动到 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Dense_part = dense_model.state_dict()


print(dense_model)

moe_model = VisionTransformer_zeke(embed_dim=192, num_heads=3)

moe_part = moe_model.state_dict()

print(moe_part)