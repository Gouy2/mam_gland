import torch
from sklearn.metrics import roc_auc_score
import os
import logging
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy import stats

from models.model_factory import ModelFactory
from config.config import MODEL_CONFIG


def train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=None):
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
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss /= len(val_loader)
    metrics, ci = calculate_metrics(np.array(all_labels), np.array(all_probs))
    
    return val_loss, metrics, ci

def calculate_metrics(y_true, y_pred_prob, confidence=0.95):
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # 计算基础指标
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics_raw = {
        'ACC': (tp + tn) / (tp + tn + fp + fn) ,
        'SENS': tp / (tp + fn)  if (tp + fn) > 0 else 0,
        'SPEC': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'PPV': tp / (tp + fp)  if (tp + fp) > 0 else 0,
        'NPV': tn / (tn + fn)  if (tn + fn) > 0 else 0,
        'AUC': roc_auc_score(y_true, y_pred_prob)
    }
    
    # 计算置信区间
    n = len(y_true)
    z = stats.norm.ppf((1 + confidence) / 2)
    
    ci = {}
    metrics = {}
    
    for metric, value in metrics_raw.items():
        if metric != 'AUC':
            # 对非AUC指标计算置信区间（使用原始比例）
            se = np.sqrt((value * (1 - value)) / n)
            ci_lower = max(0, value - z * se)
            ci_upper = min(1, value + z * se)
            
            # 转换为百分比
            metrics[metric] = value * 100
            ci[metric] = [ci_lower * 100, ci_upper * 100]
        else:
            # AUC保持原样
            se = np.sqrt((value * (1 - value)) / n)
            metrics[metric] = value
            ci[metric] = [max(0, value - z * se), min(1, value + z * se)]
    
    return metrics, ci
    

def create_experiment_dirs(base_time):
    base_dir = f'./results/{base_time}'
    dirs = {
        'log': os.path.join(base_dir, 'logs'),
        'model': os.path.join(base_dir, 'models'),
        'plot': os.path.join(base_dir, 'plots')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def train(full_dataset, hyper_params):
    
    k_folds = hyper_params['k_folds']
    batch_size = hyper_params['batch_size']
    num_epochs = hyper_params['num_epochs']
    lr = hyper_params['lr']
    weight_decay = hyper_params['weight_decay']

    # 获取当前时间，并格式化为字符串
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    dirs = create_experiment_dirs(current_time)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # 创建一个文件处理器，用于写入日志文件
    file_handler = logging.FileHandler(os.path.join(dirs['log'], 'train.log'))
    file_handler.setLevel(logging.INFO)

    # 创建一个控制台处理器，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Start print log")
    logger.info(f"Model: {MODEL_CONFIG['model_name']} , use_cbam: {MODEL_CONFIG['use_cbam']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL_CONFIG['input_channels'] == 2:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize([0.5, 0.5], [0.5, 0.5])
        ])

        # 验证集保持原始数据
        val_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),  # 与训练集一致的归一化
        ])

    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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
        model = ModelFactory.get_model(
            MODEL_CONFIG['model_name'],
            num_classes=MODEL_CONFIG['num_classes'],
            input_channels=MODEL_CONFIG['input_channels'],
            use_cbam=MODEL_CONFIG['use_cbam']
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
        
        train_losses = []
        val_losses = []

        best_model_metrics = None
        best_model_ci = None
        best_score = 0.6  # 用于判断最优模型
        best_epoch = 0

        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # 验证
            val_loss, metrics, ci = validate(model, val_loader, criterion, device)

            current_score = 0.5 * metrics['AUC'] + 0.5 * metrics['ACC']

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if current_score > best_score:
                best_score = current_score
                best_model_metrics = metrics
                best_model_ci = ci
                best_epoch = epoch
                torch.save(model.state_dict(), 
                          os.path.join(dirs['model'], f'best_model_fold_{fold+1}_epoch_{epoch+1}.pth'))

            # 记录每个epoch的指标
            logger.info(f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, '
                       f'Val Loss={val_loss:.4f}, '
                       f'AUC={metrics["AUC"]:.4f}, '
                       f'ACC={metrics["ACC"]:.2f}, '                      
                       f'SENS={metrics["SENS"]:.2f}, '
                       f'SPEC={metrics["SPEC"]:.2f}, '
                       f'PPV={metrics["PPV"]:.2f}, '
                       f'NPV={metrics["NPV"]:.2f}')

        
        # 在每个fold结束时输出最优模型的指标和置信区间
        logger.info(f'\nFold {fold + 1} Best Model Metrics:')
        logger.info(f'Best Epoch: {best_epoch + 1}')
        for metric in ['AUC', 'ACC', 'SENS', 'SPEC', 'PPV', 'NPV']:
            if best_model_metrics is None:
                logger.info('Best Model: None')
            else:
                if metric == 'AUC':
                    logger.info(f'{metric}: {best_model_metrics[metric]:.4f} '
                        f'[{best_model_ci[metric][0]:.4f}, {best_model_ci[metric][1]:.4f}]')
                else:
                    logger.info(f'{metric}: {best_model_metrics[metric]:.2f} '
                        f'[{best_model_ci[metric][0]:.2f}, {best_model_ci[metric][1]:.2f}]')

        # 绘制并保存Loss曲线
        plt.figure(figsize=(8, 6))
        plt.plot(range(num_epochs), train_losses, label='Train Loss', color='blue')
        plt.plot(range(num_epochs), val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold + 1} Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(dirs['plot'], f'fold_{fold + 1}_loss_curve.png'))
        plt.close()

def test(test_dataset, model_path, num_classes=2, input_channels=2):
    # 获取当前时间，并格式化为字符串
    # current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # dirs = create_experiment_dirs(current_time)

    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler('test.log')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Start test log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置数据转换
    if input_channels == 2:
        test_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
        ])
    else:
        test_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    # 应用转换
    test_dataset.transform = test_transform
    
    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 初始化模型
    model = ModelFactory.get_model(
        MODEL_CONFIG['model_name'],
        num_classes=MODEL_CONFIG['num_classes'],
        input_channels=MODEL_CONFIG['input_channels'],
        use_cbam=MODEL_CONFIG['use_cbam']
    ).to(device)
    
    # 加载预训练模型
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # 测试
    test_loss, metrics, ci = validate(model, test_loader, criterion, device)

    # 输出测试结果
    logger.info('\nTest Results:')
    logger.info(f'Test Loss: {test_loss:.4f}')
    for metric in ['AUC', 'ACC', 'SENS', 'SPEC', 'PPV', 'NPV']:
        if metric == 'AUC':
            logger.info(f'{metric}: {metrics[metric]:.4f} [{ci[metric][0]:.4f}, {ci[metric][1]:.4f}]')
        else:
            logger.info(f'{metric}: {metrics[metric]:.2f} [{ci[metric][0]:.2f}, {ci[metric][1]:.2f}]')

    return metrics, ci

