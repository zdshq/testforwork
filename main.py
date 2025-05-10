import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import time
from tqdm import tqdm

from model.vision_transformer_zeke import VisionTransformer_zeke

def main():
    # 超参数设置
    num_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    image_size = 224

    # 数据增强及归一化
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # CIFAR100 的均值和标准差，可根据需要微调
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])

    # CIFAR100 数据集加载
    train_dataset = datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root='../data', train=False, download=True, transform=test_transform)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型、损失函数和优化器
    model = VisionTransformer_zeke(embed_dim=192, num_heads=3, num_classes=100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # train_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        train_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, targets) in enumerate(train_bar):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {elapsed:.2f}s, Avg Loss: {running_loss/len(train_loader):.4f}")

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
        print(f"Test Accuracy after epoch {epoch+1}: {100 * correct / total:.2f}%\n")

if __name__ == '__main__':
    main()