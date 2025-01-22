import torch
from sklearn.metrics import roc_auc_score
import os
import logging
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from module.model import Resnet_cbam
from torchvision import transforms
from datetime import datetime


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    
    return val_loss, val_acc, auc

def save_best_model(model, val_loss, auc, best_val_loss, best_auc, fold, epoch, save_dir='./model_save'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存最好的损失模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'{save_dir}/best_loss_{fold + 1}_{epoch + 1}.pth')
    
    # 保存最好的AUC模型
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), f'{save_dir}/best_auc_{fold + 1}_{epoch + 1}.pth')
    
    return best_val_loss, best_auc


def train(full_dataset,num_classes = 2,input_channels = 2,
          k_folds = 5,batch_size = 4,
          num_epochs = 30,
          lr = 1e-3,
          weight_decay = 1e-4):
    
    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # 移除所有已存在的处理器
    logger.handlers.clear()

    # 创建文件处理器
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)

    # 创建控制台处理器
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # 将处理器添加到日志记录器
    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info("Start print log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 获取当前时间，并格式化为字符串
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 创建以时间命名的主文件夹
    base_dir = f'./image/{current_time}'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if input_channels == 2:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            # transforms.RandomVerticalFlip(),    # 随机垂直翻转
            transforms.RandomRotation(30),      # 随机旋转 
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
            transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),  # 归一化到 [-1, 1]
        ])

        # 验证集保持原始数据
        val_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),  # 与训练集一致的归一化
        ])

    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            # transforms.RandomVerticalFlip(),    # 随机垂直翻转
            transforms.RandomRotation(30),      # 随机旋转 
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # 归一化到 [-1, 1]
        ])

        # 验证集保持原始数据
        val_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # 与训练集一致的归一化
        ])


    # 定义 KFold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):

        logger.info(f'Fold {fold + 1}/{k_folds}')
        
        # 创建子集
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        # 添加数据增强
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform
        
        # 创建 DataLoader
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

        # for batch_idx, (inputs, targets) in enumerate(train_loader):
        #     print(f"Batch {batch_idx + 1}")
        #     print(f"Input shape: {inputs.shape}")
        #     print(f"Target shape: {targets.shape}")
        #     break 
        
        # 初始化模型、损失函数和优化器
        model = Resnet_cbam(num_classes = num_classes, input_channels=input_channels).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
        
        best_val_loss = float('inf')
        best_auc = 0.7
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # 验证
            val_loss, val_acc, auc = validate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # 打印并记录信息
            logger.info(f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, '
                         f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, AUC={auc:.4f}')
            
            # 保存最佳模型
            best_val_loss, best_auc = save_best_model(model, val_loss, auc, best_val_loss, best_auc, fold, epoch)
        
        logger.info(f'Fold {fold + 1} 完成，最佳 Val Loss: {best_val_loss:.4f}, 最佳 AUC : {best_auc:.4f}\n')

        # 绘制并保存Loss曲线
        plt.figure(figsize=(8, 6))
        plt.plot(range(num_epochs), train_losses, label='Train Loss', color='blue')
        plt.plot(range(num_epochs), val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold + 1} Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(base_dir, f'fold_{fold + 1}_loss.png'))
        plt.close()