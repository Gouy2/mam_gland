import torch
import torch.nn as nn
import torchvision.models as models
from models.cbam import CBAM

# ResNet18 模型
class Resnet18_cbam(nn.Module):
    def __init__(self, num_classes=2, input_channels=2, use_cbam=True):
        super(Resnet18_cbam, self).__init__()
        # 使用预训练的 ResNet18 模型
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.use_cbam = use_cbam
        
        # 在每个残差块后添加 CBAM 注意力模块
        self.cbam1 = CBAM(64)   
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        
        # 替换最后的全连接层，用于4分类
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.base_model.fc.in_features, num_classes)  # 四分类
            # nn.Linear(self.base_model.fc.in_features, 1)  # 二分类
        )

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        if self.use_cbam:
        
            x = self.base_model.layer1(x)
            x = self.cbam1(x)
            x = self.base_model.layer2(x)
            x = self.cbam2(x)
            x = self.base_model.layer3(x)
            x = self.cbam3(x)
            x = self.base_model.layer4(x)
            x = self.cbam4(x)

        else:
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x


# ResNet50 模型
class Resnet50_cbam(nn.Module):
    def __init__(self, num_classes=2, input_channels=2, use_cbam=True):
        super(Resnet50_cbam, self).__init__()
        
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.use_cbam = use_cbam
        
        # 在每个残差块后添加 CBAM 注意力模块
        self.cbam1 = CBAM(256) 
        self.cbam2 = CBAM(512)   
        self.cbam3 = CBAM(1024) 
        self.cbam4 = CBAM(2048) 
        
        # 替换最后的全连接层，用于4分类
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.base_model.fc.in_features, num_classes)  # 四分类
            # nn.Linear(self.base_model.fc.in_features, 1)  # 二分类
        )

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        if self.use_cbam:        
            x = self.base_model.layer1(x)
            x = self.cbam1(x)
            x = self.base_model.layer2(x)
            x = self.cbam2(x)
            x = self.base_model.layer3(x)
            x = self.cbam3(x)
            x = self.base_model.layer4(x)
            x = self.cbam4(x)

        else:
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x


# ResNet101 模型
class Resnet101_cbam(nn.Module):
    def __init__(self, num_classes=2, input_channels=2, use_cbam=True):
        super(Resnet101_cbam, self).__init__()
        
        self.base_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.use_cbam = use_cbam
        
        # 在每个残差块后添加 CBAM 注意力模块
        self.cbam1 = CBAM(256) 
        self.cbam2 = CBAM(512)   
        self.cbam3 = CBAM(1024) 
        self.cbam4 = CBAM(2048) 
        
        # 替换最后的全连接层，用于4分类
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.base_model.fc.in_features, num_classes)  # 四分类
            # nn.Linear(self.base_model.fc.in_features, 1)  # 二分类
        )

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        if self.use_cbam:        
            x = self.base_model.layer1(x)
            x = self.cbam1(x)
            x = self.base_model.layer2(x)
            x = self.cbam2(x)
            x = self.base_model.layer3(x)
            x = self.cbam3(x)
            x = self.base_model.layer4(x)
            x = self.cbam4(x)

        else:
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x
