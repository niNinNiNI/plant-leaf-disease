"""
数据加载与预处理
"""
import os
import numpy as np
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from src.transforms import get_train_transform, get_eval_transform


class PlantVillageDataset(Dataset):
    """
    PlantVillage 自定义数据集类

    参数:
        root_dir: 数据集根目录路径
        transform: torchvision transforms 组合
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # 遍历所有类别文件夹
        classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            self.idx_to_class[idx] = cls

            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue

            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                    self.samples.append((os.path.join(cls_dir, img_name), idx))

        print(f"数据集加载完成: {len(self.samples)} 张图像, {len(classes)} 个类别")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        """获取类别分布"""
        return Counter([label for _, label in self.samples])


def split_dataset(data_root, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                  seed=42, image_size=224):
    """
    划分数据集为训练集、验证集、测试集（分层抽样）
    """
    # 分别创建带增强和不带增强的数据集
    train_transform = get_train_transform(image_size=image_size)
    eval_transform = get_eval_transform(image_size=image_size)

    train_data = PlantVillageDataset(data_root, transform=train_transform)
    eval_data = PlantVillageDataset(data_root, transform=eval_transform)

    # 获取所有标签用于分层抽样
    all_labels = [label for _, label in train_data.samples]

    # 计算各子集大小
    total = len(train_data)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - val_size - test_size

    indices = np.arange(total)

    # 第一次划分：分出测试集
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size,
        stratify=all_labels, random_state=seed
    )

    # 第二次划分：从训练+验证中分出验证集
    train_val_labels = [all_labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=seed
    )

    # 创建子集
    train_dataset = Subset(train_data, train_idx)
    val_dataset = Subset(eval_data, val_idx)
    test_dataset = Subset(eval_data, test_idx)

    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_dataset)} 张 ({len(train_dataset)/total*100:.1f}%)")
    print(f"  验证集: {len(val_dataset)} 张 ({len(val_dataset)/total*100:.1f}%)")
    print(f"  测试集: {len(test_dataset)} 张 ({len(test_dataset)/total*100:.1f}%)")

    # 验证各子集中类别分布
    for name, subset in [("训练集", train_dataset), ("验证集", val_dataset), ("测试集", test_dataset)]:
        labels = [eval_data.samples[i][1] for i in subset.indices]
        dist = Counter(labels)
        print(f"  {name}类别分布: min={min(dist.values())}, max={max(dist.values())}, "
              f"avg={np.mean(list(dist.values())):.0f}")

    return train_dataset, val_dataset, test_dataset, eval_data.class_to_idx, eval_data


def _get_labels_from_dataset(dataset):
    """从 dataset 或 Subset 中提取标签列表"""
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        # Subset 类型
        return [dataset.dataset.samples[i][1] for i in dataset.indices]
    else:
        # 完整 Dataset 类型
        return [label for _, label in dataset.samples]


def compute_class_weights(dataset, num_classes):
    """计算各类别的损失权重（inverse frequency）"""
    labels = _get_labels_from_dataset(dataset)
    counts = np.bincount(labels, minlength=num_classes)

    total = len(labels)
    weights = total / (num_classes * counts)
    # 对极端权重进行平滑
    weights = np.clip(weights, 0.5, 5.0)

    return torch.tensor(weights, dtype=torch.float32)


def create_balanced_sampler(dataset, num_classes):
    """创建加权随机采样器（处理 Subset 和普通 Dataset）"""
    labels = _get_labels_from_dataset(dataset)
    counts = np.bincount(labels, minlength=num_classes)

    sample_weights = [1.0 / counts[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler


def create_dataloaders(train_dataset, val_dataset, test_dataset,
                       num_classes, batch_size=32, num_workers=4):
    """创建 DataLoader（训练集使用加权采样）"""

    balanced_sampler = create_balanced_sampler(train_dataset, num_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=balanced_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"DataLoader 创建完成:")
    print(f"  训练批次数: {len(train_loader)} (batch_size={batch_size})")
    print(f"  验证批次数: {len(val_loader)}")
    print(f"  测试批次数: {len(test_loader)}")

    return train_loader, val_loader, test_loader
