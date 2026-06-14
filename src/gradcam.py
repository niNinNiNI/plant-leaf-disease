"""
Grad-CAM 可解释性分析
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM 实现 — 使用 hook 捕获激活值和梯度

    原理: 对最后一个卷积层的输出做反向传播，
          用梯度加权激活图得到热力图
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Forward hook: capture activations
        def forward_hook(module, inp, outp):
            self.activations = outp

        # Backward hook: capture gradients of activations
        def backward_hook(module, grad_inp, grad_outp):
            self.gradients = grad_outp[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_image, target_class=None):
        """
        生成 Grad-CAM 热力图
        """
        # Temporarily enable grad for target layer params so gradients flow
        orig_requires_grad = {}
        for name, param in self.target_layer.named_parameters():
            orig_requires_grad[name] = param.requires_grad
            param.requires_grad_(True)

        self.gradients = None
        self.activations = None

        # Forward
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(1).item()

        # Backward
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=False)

        # Restore original requires_grad
        for name, param in self.target_layer.named_parameters():
            param.requires_grad_(orig_requires_grad[name])

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                f"GradCAM failed: gradients={self.gradients is not None}, "
                f"activations={self.activations is not None}"
            )

        # Compute Grad-CAM: global average pool of gradients = weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # 上采样到原始图像尺寸
        cam = F.interpolate(cam, size=input_image.shape[2:],
                            mode='bilinear', align_corners=False)

        # 归一化
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def get_target_layer(model, model_type='transfer', backbone='resnet50'):
    """获取模型的最后一个卷积层（用于 Grad-CAM）"""
    if model_type == 'basic':
        return model.features[-1].conv
    elif model_type == 'improved':
        return model.layer4[-1].conv2
    elif model_type == 'transfer':
        if backbone.startswith('resnet'):
            return model.backbone.layer4[-1].conv3
        elif backbone.startswith('efficientnet'):
            return model.backbone.features[-1][0]
        elif backbone.startswith('densenet'):
            try:
                return model.backbone.features.norm5
            except:
                return model.backbone.features[-1]
        elif backbone.startswith('mobilenet'):
            return model.backbone.features[-1][0]
        else:
            # 默认: 尝试找到最后一个 Conv2d 层
            last_conv = None
            for module in model.backbone.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            return last_conv


def visualize_gradcam(model, test_dataset, device, class_names,
                      model_type='transfer', backbone='resnet50',
                      num_samples=9, target_classes=None,
                      save_dir='results/figures'):
    """使用 Grad-CAM 可视化模型关注的区域"""
    model.eval()

    # 获取目标层
    target_layer = get_target_layer(model, model_type, backbone)
    gradcam = GradCAM(model, target_layer)

    # 选择样本
    if target_classes is not None:
        selected_indices = []
        for cls in target_classes:
            cls_indices = [i for i in range(len(test_dataset))
                           if test_dataset[i][1] == cls]
            if cls_indices:
                selected_indices.append(np.random.choice(cls_indices))
        selected_indices = selected_indices[:num_samples]
    else:
        selected_indices = np.random.choice(
            len(test_dataset),
            min(num_samples, len(test_dataset)),
            replace=False
        )

    n = len(selected_indices)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1:
        axes = axes
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx, dataset_idx in enumerate(selected_indices):
        image, true_label = test_dataset[dataset_idx]
        input_image = image.unsqueeze(0).to(device)

        # 模型预测
        with torch.no_grad():
            output = model(input_image)
            pred_label = output.argmax(1).item()
            pred_prob = F.softmax(output, dim=1)[0][pred_label].item()

        # 生成热力图（对预测类别）
        heatmap = gradcam.generate(input_image, target_class=pred_label)

        # 原始图像（反归一化）
        original = image.cpu().numpy().transpose(1, 2, 0)
        original = original * std + mean
        original = np.clip(original, 0, 1)

        # 叠加热力图
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

        superimposed = 0.6 * original + 0.4 * heatmap_colored
        superimposed = np.clip(superimposed, 0, 1)

        # 显示
        axes[idx].imshow(superimposed)

        true_name = class_names[true_label][:25]
        pred_name = class_names[pred_label][:25]
        color = 'green' if true_label == pred_label else 'red'

        axes[idx].set_title(
            f"True: {true_name}\nPred: {pred_name} ({pred_prob:.2f})",
            color=color, fontsize=9
        )
        axes[idx].axis('off')

    # 隐藏多余子图
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Grad-CAM: 模型关注区域可视化', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'gradcam_visualization.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Grad-CAM 可视化完成，已保存到 {save_dir}/gradcam_visualization.png")
    print(f"\n分析提示:")
    print(f"  - 绿色标题 = 分类正确, 红色标题 = 分类错误")
    print(f"  - 红色热力区 = 模型关注区域")
    print(f"  - 如果热力区集中在背景，说明模型发生了背景过拟合")
    print(f"  - 如果热力区集中在病斑区域，说明模型学到了正确特征")
