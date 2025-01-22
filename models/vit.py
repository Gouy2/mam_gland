import torch
import torch.nn as nn
import torchvision.models as models
import torch.hub
from timm import create_model
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class vit_b_16(nn.Module):
    def __init__(self, num_classes=2, input_channels=2, use_cbam=True):
        super(vit_b_16, self).__init__()
        
        # 使用预训练的ViT模型
        # self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit = models.vit_b_16()

        # 保存原始投影层的权重
        # 在新版本中，投影层的位置发生了变化
        original_proj = self.vit.conv_proj
        
        # 创建新的投影层，同时保持原始权重
        if input_channels != 3:  # 3是原始输入通道数
            new_proj = nn.Conv2d(
                in_channels=input_channels,
                out_channels=768,  # ViT-B/16的隐藏维度
                kernel_size=16,
                stride=16
            )
            
            # 初始化新投影层的权重
            with torch.no_grad():
                # 如果输入通道数小于3，取平均值
                if input_channels < 3:
                    new_proj.weight.data = original_proj.weight.data[:, :input_channels, :, :].clone()
                # 如果输入通道数大于3，复制并平均分配
                else:
                    new_proj.weight.data = torch.repeat_interleave(
                        original_proj.weight.data, 
                        repeats=input_channels // 3 + 1, 
                        dim=1
                    )[:, :input_channels, :, :]
                
                if original_proj.bias is not None:
                    new_proj.bias = nn.Parameter(original_proj.bias.clone())
            
            self.vit.conv_proj = new_proj
        
        # 修改分类头以适应类别数
        self.vit.heads = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        return self.vit(x)
    

class DeiT(nn.Module):
    def __init__(self, num_classes=2, input_channels=2, use_cbam=True):
        super(DeiT, self).__init__()
        
        # 创建DeiT模型
        self.deit = create_model('deit_small_patch16_224', pretrained=False)
        
        # 修改第一层以适应输入通道数
        self.deit.patch_embed.proj = nn.Conv2d(input_channels, 
                                              self.deit.patch_embed.proj.out_channels,
                                              kernel_size=self.deit.patch_embed.proj.kernel_size,
                                              stride=self.deit.patch_embed.proj.stride,
                                              padding=self.deit.patch_embed.proj.padding)
        
        # 修改分类头以适应目标类别数
        self.deit.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.deit.head.in_features, num_classes)
        )
        
        # 如果使用了蒸馏头，也需要修改
        if hasattr(self.deit, 'head_dist'):
            self.deit.head_dist = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(self.deit.head.in_features, num_classes)
            )

    def forward(self, x):
        # DeiT期望的输入尺寸为 [batch_size, channels, 224, 224]
        # 如果输入尺寸不是224x224，可能需要调整
        x = self.deit(x)
        
        # 如果模型返回元组（在训练时可能包含蒸馏token的输出），只返回主分类头的输出
        if isinstance(x, tuple):
            x = x[0]
            
        return x
    

class MobileViT(nn.Module):
    def __init__(self, num_classes=2, input_channels=2, use_cbam=True):
        super(MobileViT, self).__init__()
        
        # 创建MobileViT模型
        self.mobilevit = create_model(
            'mobilevit_xs', 
            pretrained=False,
            img_size=224,  # 指定输入尺寸
            num_classes=num_classes
        )
        
        # 修改第一层以适应输入通道数
        # MobileViT使用stem作为第一层
        
        self.mobilevit.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                16,  # 原始输出通道数是16
                kernel_size=(3, 3),  # 原始kernel_size
                stride=(2, 2),       # 原始stride
                padding=(1, 1),      # 原始padding
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True)    # 使用SiLU激活函数，与原始模型一致
        )
        
        # 修改分类头
        in_features = self.mobilevit.head.fc.in_features
        self.mobilevit.head.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.mobilevit(x)