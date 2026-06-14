"""
模型定义：BasicCNN / ImprovedCNN / TransferLearningModel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# ConvBlock — 基础卷积块
# ============================================================
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool → Dropout2d"""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 dropout_p=0.0, use_pool=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if use_pool else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


# ============================================================
# BasicCNN — 从零搭建的基础 CNN
# ============================================================
class BasicCNN(nn.Module):
    """
    基础 CNN: 5 个 ConvBlock + GAP + 2 层 FC

    结构:
        ConvBlock(3→32) → ConvBlock(32→64) → ConvBlock(64→128)
        → ConvBlock(128→256) → ConvBlock(256→512) → GAP → FC(512→256) → FC(256→num_classes)
    """

    def __init__(self, num_classes=15, dropout_rate=0.5):
        super(BasicCNN, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout_p=0.1),
            ConvBlock(32, 64, dropout_p=0.15),
            ConvBlock(64, 128, dropout_p=0.2),
            ConvBlock(128, 256, dropout_p=0.25),
            ConvBlock(256, 512, dropout_p=0.3, use_pool=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        """提取特征向量，用于 Grad-CAM 可视化"""
        return self.features(x)


# ============================================================
# ResidualBlock — 残差块
# ============================================================
class ResidualBlock(nn.Module):
    """残差块，参考 ResNet 设计"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# ============================================================
# ImprovedCNN — 改进版 CNN（残差连接）
# ============================================================
class ImprovedCNN(nn.Module):
    """
    改进的 CNN：引入残差连接

    相比 BasicCNN 的优势：
    - 残差连接缓解梯度消失
    - 更深的网络可训练
    """

    def __init__(self, num_classes=15, dropout_rate=0.5):
        super(ImprovedCNN, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# ============================================================
# TransferLearningModel — 迁移学习模型
# ============================================================
class TransferLearningModel(nn.Module):
    """
    基于预训练模型的迁移学习

    支持的骨干网络:
    - resnet50, resnet101
    - efficientnet_b0, efficientnet_b3
    - densenet121, densenet169
    - mobilenet_v3_large
    - convnext_tiny
    """

    BACKBONES = {
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, 2048),
        'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1, 2048),
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
        'efficientnet_b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1, 1536),
        'densenet121': (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, 1024),
        'densenet169': (models.densenet169, models.DenseNet169_Weights.IMAGENET1K_V1, 1664),
        'mobilenet_v3': (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1, 960),
        'convnext_tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
    }

    def __init__(self, backbone_name='resnet50', num_classes=15,
                 dropout_rate=0.5, freeze_backbone=True):
        super(TransferLearningModel, self).__init__()

        if backbone_name not in self.BACKBONES:
            raise ValueError(f"不支持的骨干网络: {backbone_name}. "
                             f"可选: {list(self.BACKBONES.keys())}")

        model_fn, weights, in_features = self.BACKBONES[backbone_name]

        self.backbone = model_fn(weights=weights)

        # 冻结骨干网络
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 替换分类头
        self._replace_classifier(backbone_name, in_features, num_classes, dropout_rate)

        self.backbone_name = backbone_name
        self.num_classes = num_classes

    def _replace_classifier(self, backbone_name, in_features, num_classes, dropout_rate):
        """替换各模型的分类头"""
        new_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

        if backbone_name.startswith('resnet') or backbone_name.startswith('resnext'):
            self.backbone.fc = new_classifier
        elif backbone_name.startswith('efficientnet'):
            self.backbone.classifier = new_classifier
        elif backbone_name.startswith('densenet'):
            self.backbone.classifier = new_classifier
        elif backbone_name.startswith('mobilenet'):
            self.backbone.classifier = new_classifier
        elif backbone_name.startswith('convnext'):
            self.backbone.classifier = new_classifier
        else:
            self.backbone.fc = new_classifier

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self, num_layers_to_unfreeze=None):
        """
        逐步解冻骨干网络用于微调

        参数:
            num_layers_to_unfreeze: 解冻最后 N 层。None 表示全部解冻。
        """
        if num_layers_to_unfreeze is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            params = list(self.backbone.parameters())
            for param in params[:-num_layers_to_unfreeze]:
                param.requires_grad = False
            for param in params[-num_layers_to_unfreeze:]:
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"骨干网络: {trainable:,}/{total:,} 参数可训练 ({trainable/total*100:.1f}%)")


# ============================================================
# 模型工厂函数
# ============================================================
def create_model(model_type='transfer', backbone='resnet50', num_classes=15,
                 dropout_rate=0.5, freeze_backbone=True):
    """
    模型工厂函数

    参数:
        model_type: 'basic' | 'improved' | 'transfer'
        backbone: 迁移学习使用的骨干网络名称
        num_classes: 分类类别数
        dropout_rate: Dropout 比例
        freeze_backbone: 是否冻结骨干网络
    """
    if model_type == 'basic':
        return BasicCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type == 'improved':
        return ImprovedCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type == 'transfer':
        return TransferLearningModel(
            backbone_name=backbone,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")
