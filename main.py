import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import time
import timm
from tqdm import tqdm
from model.vision_transformer_zeke import VisionTransformer_zeke
import os
import datetime
import json
from pathlib import Path

def main():

    # 超参数设置
    num_epochs = 300
    batch_size = 128
    initial_learning_rate = 0.001 # This will be the peak LR after warmup
    image_size = 224
    warmup_epochs = 10 # Number of epochs for learning rate warmup
    weight_decay = 0.05 # Significantly reduced, common for ViTs with AdamW
    gradient_clipping_norm = 1.0 # Common value for gradient clipping
    num_experts = 192
    # 创建基于时间的运行ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_{num_experts}"
    
    # 创建result目录以及本次运行的子目录
    result_dir = Path("result")
    run_dir = result_dir / run_name
    
    # 确保目录存在
    result_dir.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=True)
    
    # 创建日志文件
    log_file = run_dir / "training_log.txt"
    model_dir = run_dir / "checkpoints"
    model_dir.mkdir(exist_ok=True)
    
    def log_message(message:str):
        print(message)
        with open(log_file, "a") as f:
            f.write(message+'\n')
    

    
    # 保存配置信息
    config = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "initial_learning_rate": initial_learning_rate,
        "image_size": image_size,
        "warmup_epochs": warmup_epochs,
        "weight_decay": weight_decay,
        "gradient_clipping_norm": gradient_clipping_norm,
        "model_type": "VisionTransformer_zeke",
        "model_params": {
            "num_heads": 3,
            "num_classes": 100,
            "depth": 12,
            "embed_dim": 4,
            "num_experts":num_experts
        },
        "pretrained_weights": "moe_model_with_pretrained_weights.pth"
    }
    
    # 将配置保存到JSON文件
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # 数据增强及归一化
    train_transform = transforms.Compose([
        transforms.Resize(image_size), # Resize to 224 first
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)), # More robust crop
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), # A good default for stronger augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size), # Ensure test images are consistently 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])

    # CIFAR100 数据集加载
    train_dataset = datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root='../data', train=False, download=True, transform=test_transform)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型、损失函数和优化器
    model = VisionTransformer_zeke(num_heads=3,num_classes=100,depth=12,embed_dim=192, num_experts=num_experts)

    
    log_message("Loading pretrained weights...")
    pretrained_weights = torch.load("moe_model_with_pretrained_weights.pth")
    model.load_state_dict(pretrained_weights)

    log_message("Successfully loaded pretrained weights!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing can help
    optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler with warmup
    # Scheduler 1: Linear warmup
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs) if epoch < warmup_epochs else 1.0
    )
    # Scheduler 2: Cosine annealing after warmup
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(num_epochs - warmup_epochs), # T_max is the number of epochs for cosine decay
        eta_min=1e-6 # Minimum learning rate
    )

    # 记录训练信息
    log_message(f"Training ViT tiny from scratch on CIFAR100 for {num_epochs} epochs.")
    log_message(f"Device: {device}")
    log_message(f"Batch size: {batch_size}, Initial LR: {initial_learning_rate}, Weight Decay: {weight_decay}")
    log_message(f"Warmup epochs: {warmup_epochs}, Grad Clip: {gradient_clipping_norm}")
    log_message(f"Results will be saved in: {run_dir}")

    # 记录最佳准确率
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        # Manually adjust LR for warmup phase, then let cosine scheduler take over
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            pass # Cosine scheduler will be stepped at the end of the epoch

        train_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, targets) in enumerate(train_bar):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            if gradient_clipping_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)
            optimizer.step()
            running_loss += loss.item()

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            train_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.1e}")

        # Step the schedulers
        if epoch < warmup_epochs:
            pass
        else:
            cosine_scheduler.step()

        elapsed = time.time() - start_time
        avg_loss = running_loss / len(train_loader)
        current_lr_end_epoch = optimizer.param_groups[0]['lr'] # Get LR at end of epoch
        
        # 记录训练信息
        log_message(f"Epoch {epoch+1} finished in {elapsed:.2f}s, Avg Loss: {avg_loss:.4f}, LR: {current_lr_end_epoch:.1e}")

        # 测试阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        log_message(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}%\n")
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'loss': avg_loss
        }

    # # 假设 model 和 optimizer 已经定义好了
    # checkpoint_path = model_dir / "checkpoint_latest.pth"
    # if checkpoint_path.exists():
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     best_accuracy = checkpoint.get('accuracy', 0) # 使用 .get 以防旧的checkpoint没有accuracy
    #     print(f"Loaded checkpoint from epoch {start_epoch-1}, resuming.")
    # else:
    #     start_epoch = 0
    #     best_accuracy = 0
    #     print("No checkpoint found, starting from scratch.")

    # # ... 然后开始你的训练循环 from start_epoch ...
    # for epoch in range(start_epoch, num_epochs):
    #     # ... training logic ...
        # 保存最新的模型
        # torch.save(checkpoint, model_dir / f"checkpoint_latest.pth")
        
        # 如果达到新的最佳准确率，则保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(checkpoint, model_dir / f"checkpoint_best.pth")
            log_message(f"New best accuracy: {best_accuracy:.2f}%, model saved!")
        
        # # 每10个epoch保存一次
        # if (epoch + 1) % 10 == 0:
        #     torch.save(checkpoint, model_dir / f"checkpoint_epoch{epoch+1}.pth")
    
    # 训练结束，记录最终信息
    log_message(f"Training completed! Best accuracy: {best_accuracy:.2f}%")
    log_message(f"All results saved in: {run_dir}")

if __name__ == '__main__':
    main()