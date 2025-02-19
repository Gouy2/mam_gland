import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.cbam import CBAM

class Densenet121_cbam(nn.Module):
    def __init__(self, num_classes=2, input_channels=2, use_cbam=True):
        super(Densenet121_cbam, self).__init__()
        
        # 加载本地预训练模型
        self.base_model = models.densenet121()
        
        # 加载并检查预训练权重
        checkpoint = torch.load('models/pretrain/DenseNet121.pt', map_location='cpu', weights_only=False)
        
        # 移除 "backbone.0." 前缀
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k.replace('backbone.0.', '')
            new_state_dict[name] = v
            
        # 加载处理后的权重
        self.base_model.load_state_dict(new_state_dict, strict=False)
        
        # 智能地调整第一层卷积以适应输入通道数
        original_conv = self.base_model.features.conv0
        
        if input_channels != 3:
            # 获取原始3通道权重
            original_weight = original_conv.weight.data  # shape [64,3,7,7]
            
            # 计算跨通道均值
            mean_weight = original_weight.mean(dim=1, keepdim=True)  # shape [64,1,7,7]
            
            # 扩展为指定的输入通道数
            new_weight = mean_weight.repeat(1, input_channels, 1, 1)  # shape [64,input_channels,7,7]
            
            # 创建新的卷积层并应用计算好的权重
            self.base_model.features.conv0 = nn.Conv2d(
                input_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            self.base_model.features.conv0.weight.data = new_weight
        
        self.use_cbam = use_cbam
        
        # 修正CBAM模块的通道数
        self.cbam1 = CBAM(128)    # 修改为正确的通道数
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(512)
        self.cbam4 = CBAM(1024)
        
        # 修改分类器
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # 第一个卷积层和池化层
        x = self.base_model.features.conv0(x)
        x = self.base_model.features.norm0(x)
        x = self.base_model.features.relu0(x)
        x = self.base_model.features.pool0(x)

        # Dense Block 1
        x = self.base_model.features.denseblock1(x)
        x = self.base_model.features.transition1(x)
        if self.use_cbam:
            x = self.cbam1(x)

        # Dense Block 2
        x = self.base_model.features.denseblock2(x)
        x = self.base_model.features.transition2(x)
        if self.use_cbam:
            x = self.cbam2(x)

        # Dense Block 3
        x = self.base_model.features.denseblock3(x)
        x = self.base_model.features.transition3(x)
        if self.use_cbam:
            x = self.cbam3(x)

        # Dense Block 4
        x = self.base_model.features.denseblock4(x)
        if self.use_cbam:
            x = self.cbam4(x)

        x = self.base_model.features.norm5(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.base_model.classifier(x)
        
        return x 