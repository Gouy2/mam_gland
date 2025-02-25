import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torchvision import transforms
from torchvision.transforms import RandAugment
from config.config import MODEL_CONFIG, PARAM_CONFIG
from models.model_factory import ModelFactory
from utils.trainer import train_one_epoch, validate, create_experiment_dirs

def freeze_layers(model, except_names=None):
    """冻结除指定层以外的所有层"""
    if except_names is None:
        except_names = []
    
    for name, param in model.named_parameters():
        if not any(except_name in name for except_name in except_names):
            param.requires_grad = False
        else:
            param.requires_grad = True

def unfreeze_layers(model, layer_names=None):
    """解冻指定的层"""
    if layer_names is None:
        layer_names = []
    
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True

def unfreeze_all(model):
    """解冻所有层"""
    for param in model.parameters():
        param.requires_grad = True

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.2):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def get_layer_params(model):
    """获取不同层的参数，使用更保守的学习率"""
    params = [
        # 输入层
        {'params': model.base_model.features.conv0.parameters(), 
         'lr': 3e-5, 'weight_decay': 5e-3},
        
        # Block 1 (保持冻结)
        {'params': model.base_model.features.denseblock1.parameters(), 
         'lr': 0, 'weight_decay': 5e-3},
        
        # Block 2 (保持冻结)
        {'params': model.base_model.features.denseblock2.parameters(), 
         'lr': 0, 'weight_decay': 5e-3},
        
        # Block 3 (保持冻结)
        {'params': model.base_model.features.denseblock3.parameters(), 
         'lr': 0, 'weight_decay': 5e-3},
        
        # Block 4 (主要训练目标)
        {'params': model.base_model.features.denseblock4.parameters(), 
         'lr': 1e-4, 'weight_decay': 5e-3},
        
        # CBAM模块
        {'params': [p for n, p in model.named_parameters() if 'cbam' in n], 
         'lr': 3e-5, 'weight_decay': 5e-3},
        
        # 分类器
        {'params': model.base_model.classifier.parameters(), 
         'lr': 1e-4, 'weight_decay': 1e-2}
    ]
    return params

def train_progressive(full_dataset, hyper_params):
    k_folds = hyper_params['k_folds']
    batch_size = hyper_params['batch_size']
    num_epochs = hyper_params['num_epochs']
    
    # 创建实验目录
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    dirs = create_experiment_dirs(current_time)
    
    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(os.path.join(dirs['log'], 'train_progressive.log'))
    console_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Start progressive training")
    logger.info(f"Model: {MODEL_CONFIG['model_name']}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据增强设置
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        RandAugment(num_ops=2, magnitude=5),
        transforms.Normalize([0.5] * MODEL_CONFIG['input_channels'], 
                           [0.5] * MODEL_CONFIG['input_channels'])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize([0.5] * MODEL_CONFIG['input_channels'], 
                           [0.5] * MODEL_CONFIG['input_channels'])
    ])
    
    # K折交叉验证
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        logger.info(f'Fold {fold + 1}/{k_folds}')
        
        # 创建数据加载器
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        model = ModelFactory.get_model(
            MODEL_CONFIG['model_name'],
            num_classes=MODEL_CONFIG['num_classes'],
            input_channels=MODEL_CONFIG['input_channels'],
            use_cbam=MODEL_CONFIG['use_cbam']
        ).to(device)
        
        # 使用标签平滑
        criterion = LabelSmoothingLoss(classes=MODEL_CONFIG['num_classes'], smoothing=0.2)
        
        # 渐进式训练阶段
        training_stages = [
            {
                'name': 'Stage 1: Only Block 4',
                'epochs': 30,
                'setup': lambda m: freeze_layers(m, except_names=['denseblock4', 'classifier']),
                'lr_multiplier': 1.0
            },
            {
                'name': 'Stage 2: Block 3 & 4',
                'epochs': 15,
                'setup': lambda m: unfreeze_layers(m, ['denseblock3', 'denseblock4']),
                'lr_multiplier': 0.3
            },
            {
                'name': 'Stage 3: Fine-tune all',
                'epochs': num_epochs - 45,
                'setup': lambda m: unfreeze_all(m),
                'lr_multiplier': 0.1
            }
        ]
        
        # 早停设置
        patience = 3
        best_score = 0.6
        no_improve_count = 0
        current_epoch = 0
        
        for stage in training_stages:
            logger.info(f"\nStarting {stage['name']}")
            stage['setup'](model)
            
            params = get_layer_params(model)
            for param_group in params:
                param_group['lr'] *= stage['lr_multiplier']
            
            optimizer = optim.AdamW(params)
            
            for epoch in range(stage['epochs']):
                current_epoch += 1
                
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device, grad_clip=0.5
                )
                val_loss, metrics, ci = validate(model, val_loader, criterion, device)
                
                current_score = 0.5 * metrics['AUC'] + 0.5 * metrics['ACC']
                
                # 早停检查
                if current_score > best_score:
                    best_score = current_score
                    no_improve_count = 0
                    torch.save(model.state_dict(), 
                             os.path.join(dirs['model'], f'best_model_fold_{fold+1}_epoch_{current_epoch}.pth'))
                else:
                    no_improve_count += 1
                    # if no_improve_count >= patience:
                    #     logger.info(f"Early stopping triggered after {epoch+1} epochs in {stage['name']}")
                    #     break
                
                logger.info(f'Epoch {current_epoch}: Train Loss={train_loss:.4f}, '
                          f'Val Loss={val_loss:.4f}, '
                          f'AUC={metrics["AUC"]:.4f}, '
                          f'ACC={metrics["ACC"]:.2f}')
        
        # 输出最佳结果
        logger.info(f'\nFold {fold + 1} Best Results (Epoch {current_epoch}):')
        for metric in ['AUC', 'ACC', 'SENS', 'SPEC', 'PPV', 'NPV']:
            if metric == 'AUC':
                logger.info(f'{metric}: {metrics[metric]:.4f} '
                          f'[{ci[metric][0]:.4f}, {ci[metric][1]:.4f}]')
            else:
                logger.info(f'{metric}: {metrics[metric]:.2f} '
                          f'[{ci[metric][0]:.2f}, {ci[metric][1]:.2f}]')

from utils.load_data import load_cached_dataset, create_imgWithLabels
from utils.load_data import process_images_for_patients,cache_dataset
from utils.dataset import ImageDataset 

def prepare_dataset(train_images, test_images, train_labels_df, test_labels_df):
    # 创建图像标签对
    images_with_labels = create_imgWithLabels(train_images, train_labels_df, is_double=False, is_2cat=True)
    images_with_labels += create_imgWithLabels(test_images, test_labels_df, is_double=False, is_2cat=True)

    # 创建数据集
    dataset = ImageDataset(images_with_labels)
    # 抽出20%的数据作为测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    return torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

if __name__ == '__main__':

    import pandas as pd

    # cache_path = 'cache/train_225_nonfo.npy'
    train_images = load_cached_dataset('cache/train_225_nonfo.npy', format='npy')
    test_images = load_cached_dataset('cache/train_180_nonfo.npy', format='npy')
    print("---加载图像数据完成---")

    # combined_images = np.concatenate([train_images, test_images], axis=0)

    # 加载标签数据
    train_excel_path = './data/new_excel/chaoyang_retrospective_233.xlsx'
    test_excel_path = './data/new_excel/chaoyang_prospective_190.xlsx' 
    train_labels_df = pd.read_excel(train_excel_path)
    test_labels_df = pd.read_excel(test_excel_path)
    print("---加载标签数据完成---")
    
    # 2. 准备数据集
    train_dataset, test_dataset = prepare_dataset(train_images, test_images, train_labels_df, test_labels_df)
    
    # 获取训练参数
    hyper_params = PARAM_CONFIG[MODEL_CONFIG['model_name']]
    
    # 开始训练
    train_progressive(train_dataset, hyper_params) 