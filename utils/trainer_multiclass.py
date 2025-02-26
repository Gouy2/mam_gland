import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from models.model_factory import ModelFactory
from config.config import MODEL_CONFIG


def train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=None):
    """训练一个epoch的多分类模型"""
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

def validate_multiclass(model, val_loader, criterion, device):
    """验证多分类模型性能"""
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss /= len(val_loader)
    metrics, ci = calculate_multiclass_metrics(
        np.array(all_labels), 
        np.array(all_preds), 
        np.array(all_probs)
    )
    
    return val_loss, metrics, ci

def calculate_multiclass_metrics(y_true, y_pred, y_probs, confidence=0.95):
    """
    计算多分类评估指标
    
    Parameters:
    -----------
    y_true : 真实标签
    y_pred : 预测标签
    y_probs : 预测概率 (n_samples, n_classes)
    confidence : 置信区间范围
    
    Returns:
    --------
    metrics : 各项指标
    ci : 置信区间
    """
    n_classes = y_probs.shape[1]
    n_samples = len(y_true)
    
    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    
    # 混淆矩阵
    conf_mat = confusion_matrix(y_true, y_pred)
    
    # 各类别的精确率、召回率、F1分数
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(n_classes), zero_division=0
    )
    
    # 宏平均和加权平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # 多分类AUC（使用One-vs-Rest策略）
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    auc_scores = []
    
    for i in range(n_classes):
        if len(np.unique(y_true_bin[:, i])) > 1:  # 确保二分类情况有两个类
            try:
                auc_scores.append(roc_auc_score(y_true_bin[:, i], y_probs[:, i]))
            except ValueError:
                # 处理ROC AUC计算错误情况
                auc_scores.append(0.5)  # 默认为随机猜测水平
        else:
            auc_scores.append(0.5)  # 没有足够样本类别时设为0.5
    
    # 计算置信区间
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # 准确率的置信区间
    acc_se = np.sqrt((accuracy * (1 - accuracy)) / n_samples)
    acc_ci = [max(0, accuracy - z * acc_se), min(1, accuracy + z * acc_se)]
    
    # 组织指标和置信区间
    metrics = {
        'ACC': accuracy * 100,
        'Macro_Precision': macro_precision * 100,
        'Macro_Recall': macro_recall * 100,
        'Macro_F1': macro_f1 * 100,
        'Weighted_Precision': weighted_precision * 100,
        'Weighted_Recall': weighted_recall * 100,
        'Weighted_F1': weighted_f1 * 100,
        'AUC_Macro': np.mean(auc_scores),
        'Confusion_Matrix': conf_mat,
    }
    
    # 添加每个类别的指标
    for i in range(n_classes):
        metrics[f'Class_{i}_Precision'] = precision[i] * 100
        metrics[f'Class_{i}_Recall'] = recall[i] * 100
        metrics[f'Class_{i}_F1'] = f1[i] * 100
        metrics[f'Class_{i}_AUC'] = auc_scores[i]
    
    # 置信区间
    ci = {
        'ACC': [acc_ci[0] * 100, acc_ci[1] * 100],
        'AUC_Macro': [max(0, np.mean(auc_scores) - z * np.std(auc_scores)/np.sqrt(n_classes)), 
                       min(1, np.mean(auc_scores) + z * np.std(auc_scores)/np.sqrt(n_classes))],
    }
    
    return metrics, ci

def create_experiment_dirs(base_time):
    """创建实验目录"""
    base_dir = f'./results/{base_time}'
    dirs = {
        'log': os.path.join(base_dir, 'logs'),
        'model': os.path.join(base_dir, 'models'),
        'plot': os.path.join(base_dir, 'plots'),
        'cm': os.path.join(base_dir, 'confusion_matrices')  # 为混淆矩阵添加单独目录
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # 设置坐标轴
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在格子中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def train_multiclass(full_dataset, hyper_params, class_names=None):
    """
    训练多分类模型
    
    Parameters:
    -----------
    full_dataset : 完整数据集
    hyper_params : 超参数
    class_names : 类别名称列表，用于可视化
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(hyper_params['num_classes'])]
    
    k_folds = hyper_params['k_folds']
    batch_size = hyper_params['batch_size']
    num_epochs = hyper_params['num_epochs']
    lr = hyper_params['lr']
    weight_decay = hyper_params['weight_decay']
    num_classes = hyper_params['num_classes']

    # 获取当前时间，并格式化为字符串
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    dirs = create_experiment_dirs(current_time)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # 创建一个文件处理器，用于写入日志文件
    file_handler = logging.FileHandler(os.path.join(dirs['log'], 'train_multiclass.log'))
    file_handler.setLevel(logging.INFO)

    # 创建一个控制台处理器，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Start print log")
    logger.info(f"Model: {MODEL_CONFIG['model_name']} , use_cbam: {MODEL_CONFIG['use_cbam']} , "
                f"learning_rate: {lr} , weight_decay: {weight_decay}, num_classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义数据增强
    if MODEL_CONFIG['input_channels'] == 2:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize([0.5, 0.5], [0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    # 定义 KFold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # 存储所有折叠的最佳指标
    all_folds_best_metrics = []
    
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
        
        # 初始化模型
        model = ModelFactory.get_model(
            MODEL_CONFIG['model_name'],
            num_classes=num_classes,
            input_channels=MODEL_CONFIG['input_channels'],
            use_cbam=MODEL_CONFIG['use_cbam']
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        train_losses = []
        val_losses = []

        best_model_metrics = None
        best_model_ci = None
        best_score = 0  # 用于判断最优模型
        best_epoch = 0
        best_confusion_matrix = None

        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # 验证
            val_loss, metrics, ci = validate_multiclass(model, val_loader, criterion, device)

            # 使用准确率和宏平均F1分数的组合作为评价标准
            current_score = 0.5 * metrics['ACC']/100 + 0.5 * metrics['Macro_F1']/100

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if current_score > best_score:
                best_score = current_score
                best_model_metrics = metrics
                best_model_ci = ci
                best_epoch = epoch
                best_confusion_matrix = metrics['Confusion_Matrix']
                torch.save(model.state_dict(), 
                          os.path.join(dirs['model'], f'best_model_fold_{fold+1}_epoch_{epoch+1}.pth'))

            # 记录每个epoch的主要指标
            logger.info(f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, '
                       f'Val Loss={val_loss:.4f}, '
                       f'ACC={metrics["ACC"]:.2f}, '
                       f'Macro_F1={metrics["Macro_F1"]:.2f}, '
                       f'AUC_Macro={metrics["AUC_Macro"]:.4f}')

        # 保存最佳模型的混淆矩阵
        if best_confusion_matrix is not None:
            cm_save_path = os.path.join(dirs['cm'], f'fold_{fold+1}_best_cm.png')
            plot_confusion_matrix(best_confusion_matrix, class_names, cm_save_path)
        
        # 在每个fold结束时输出最优模型的指标和置信区间
        logger.info(f'\nFold {fold + 1} Best Model Metrics:')
        logger.info(f'Best Epoch: {best_epoch + 1}')
        
        # 输出主要性能指标
        main_metrics = ['ACC', 'Macro_F1', 'Weighted_F1', 'AUC_Macro']
        for metric in main_metrics:
            if best_model_metrics is None:
                logger.info('Best Model: None')
            else:
                if metric in best_model_ci:
                    logger.info(f'{metric}: {best_model_metrics[metric]:.2f} '
                        f'[{best_model_ci[metric][0]:.2f}, {best_model_ci[metric][1]:.2f}]')
                else:
                    logger.info(f'{metric}: {best_model_metrics[metric]:.2f}')
        
        # 输出每个类别的指标
        logger.info('\nPer-class metrics:')
        for i in range(num_classes):
            logger.info(f'Class {i} ({class_names[i]}):')
            logger.info(f'  Precision: {best_model_metrics[f"Class_{i}_Precision"]:.2f}%')
            logger.info(f'  Recall: {best_model_metrics[f"Class_{i}_Recall"]:.2f}%')
            logger.info(f'  F1: {best_model_metrics[f"Class_{i}_F1"]:.2f}%')
            logger.info(f'  AUC: {best_model_metrics[f"Class_{i}_AUC"]:.4f}')
        
        # 存储这个折叠的最佳指标
        all_folds_best_metrics.append(best_model_metrics)

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
    
    # 计算所有折叠的平均性能
    logger.info('\n=== Cross-validation Average Results ===')
    avg_metrics = {}
    
    # 提取主要指标
    for metric in main_metrics:
        metric_values = [fold_metrics[metric] for fold_metrics in all_folds_best_metrics if fold_metrics is not None]
        if metric_values:
            avg_metrics[metric] = np.mean(metric_values)
            logger.info(f'Average {metric}: {avg_metrics[metric]:.2f} ± {np.std(metric_values):.2f}')
    
    # 提取每个类别的指标
    for i in range(num_classes):
        logger.info(f'\nAverage Class {i} ({class_names[i]}) metrics:')
        for metric_type in ['Precision', 'Recall', 'F1']:
            key = f'Class_{i}_{metric_type}'
            metric_values = [fold_metrics[key] for fold_metrics in all_folds_best_metrics if fold_metrics is not None]
            if metric_values:
                avg = np.mean(metric_values)
                std = np.std(metric_values)
                logger.info(f'  {metric_type}: {avg:.2f}% ± {std:.2f}%')
        
        # AUC
        key = f'Class_{i}_AUC'
        metric_values = [fold_metrics[key] for fold_metrics in all_folds_best_metrics if fold_metrics is not None]
        if metric_values:
            avg = np.mean(metric_values)
            std = np.std(metric_values)
            logger.info(f'  AUC: {avg:.4f} ± {std:.4f}')
    
    return avg_metrics

def test_multiclass(test_dataset, model_path, class_names=None, num_classes=4, input_channels=2):
    """
    测试多分类模型
    
    Parameters:
    -----------
    test_dataset : 测试数据集
    model_path : 模型路径
    class_names : 类别名称
    num_classes : 类别数量
    input_channels : 输入通道数
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler('test_multiclass.log')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Start multiclass test log")

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
        num_classes=num_classes,
        input_channels=input_channels,
        use_cbam=MODEL_CONFIG['use_cbam']
    ).to(device)
    
    # 加载预训练模型
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # 测试
    test_loss, metrics, ci = validate_multiclass(model, test_loader, criterion, device)

    # 获取当前时间并创建测试结果目录
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_dir = f'./test_results/{current_time}'
    os.makedirs(test_dir, exist_ok=True)
    
    # 保存混淆矩阵
    cm_save_path = os.path.join(test_dir, 'test_confusion_matrix.png')
    plot_confusion_matrix(metrics['Confusion_Matrix'], class_names, cm_save_path)

    # 输出测试结果
    logger.info('\n=== Test Results ===')
    logger.info(f'Test Loss: {test_loss:.4f}')
    
    # 输出主要性能指标
    main_metrics = ['ACC', 'Macro_F1', 'Weighted_F1', 'AUC_Macro']
    for metric in main_metrics:
        if metric in ci:
            logger.info(f'{metric}: {metrics[metric]:.2f} [{ci[metric][0]:.2f}, {ci[metric][1]:.2f}]')
        else:
            logger.info(f'{metric}: {metrics[metric]:.2f}')
    
    # 输出每个类别的指标
    logger.info('\nPer-class metrics:')
    for i in range(num_classes):
        logger.info(f'Class {i} ({class_names[i]}):')
        logger.info(f'  Precision: {metrics[f"Class_{i}_Precision"]:.2f}%')
        logger.info(f'  Recall: {metrics[f"Class_{i}_Recall"]:.2f}%')
        logger.info(f'  F1: {metrics[f"Class_{i}_F1"]:.2f}%')
        logger.info(f'  AUC: {metrics[f"Class_{i}_AUC"]:.4f}')

    return metrics, ci 