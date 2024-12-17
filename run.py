import numpy as np
from pathlib import Path
import pandas as pd
import psutil
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import module.model as model
import importlib
importlib.reload(model)
from module.model import Resnet18_cbam
from module.dataset import ImageDataset

print(1)

def print_memory_usage():
    process = psutil.Process()
    print(f"内存使用: {process.memory_info().rss / (1024*1024):.2f} MB")

def load_cached_dataset(cache_path, format='npy'):
    cache_path = Path(cache_path)
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    # 检查文件大小
    file_size = os.path.getsize(cache_path)
    print(f"文件大小: {file_size / (1024*1024*1024):.2f} GB")
    
    print("开始加载前内存使用情况:")
    print_memory_usage()
    
    try:
        if format == 'npy':
            print("正在加载npy文件...")
            data = np.load(cache_path, mmap_mode='r')
            print("加载完成")
            print("加载后内存使用情况:")
            print_memory_usage()
            return data
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        raise

# 使用优化后的函数
patient_images = load_cached_dataset('cache/train.npy', format='npy')
print("完成数据缓存加载")

excel_path = './data/beiyou_excel/chaoyang_retrospective_233.xlsx'  # 包含病人姓名和标签的Excel文件路径
labels_df = pd.read_excel(excel_path)

# 补全标签并构建 images_with_labels 列表
images_with_labels = []
labels = []
for i, patient_input in enumerate(patient_images):
    label = labels_df.iloc[i]['N分期']  # 按顺序获取对应的标签

    # 如果标签为 NaN，则用均值填充
    if pd.isna(label):
        label = 1.0
    elif label == 2.0 or label == 3.0 :
        label = 1.0
    
    images_with_labels.append((patient_input, label))
    labels.append(label)

# 输出处理后的标签
# for i, (imageinput, label) in enumerate(images_with_labels):
#     print(imageinput.shape)
    # break
    # print(f"第 {i+1} 项标签: {label}")
# print(labels)

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

# 创建数据集实例
full_dataset = ImageDataset(images_with_labels)


# 假设 ImageDataset 和 Resnet18_cbam 已定义
full_dataset = ImageDataset(images_with_labels)
k_folds = 5
batch_size = 4
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备:{device}")

# 定义 KFold
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# 存储每个fold的结果
fold_results = {}

for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    print(f'Fold {fold + 1}/{k_folds}')
    
    # 创建子集
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    
    # 添加数据增强
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform
    
    # 创建 DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 初始化模型、损失函数和优化器
    model = Resnet18_cbam(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练
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
        
        # 验证
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
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, AUC={auc:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pth')
    
    fold_results[fold] = {'val_loss': best_val_loss}
    print(f'Fold {fold + 1} 完成，最佳 Val Loss: {best_val_loss:.4f}\n')

# 输出所有fold的结果
for fold in fold_results:
    print(f'Fold {fold + 1} 最佳 Val Loss: {fold_results[fold]["val_loss"]:.4f}')

