"""
全局配置参数
"""
import os

class Config:
    """全局配置"""

    # ==================== 路径 ====================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT = os.path.join(os.path.dirname(BASE_DIR), "archive/plantvillage/PlantVillage")
    # 如果上面路径不对，尝试同级目录
    if not os.path.exists(DATA_ROOT):
        DATA_ROOT = os.path.join(BASE_DIR, "archive/plantvillage/PlantVillage")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    RESULT_DIR = os.path.join(BASE_DIR, "results")
    FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
    LOG_DIR = os.path.join(RESULT_DIR, "logs")

    # ==================== 数据 ====================
    IMAGE_SIZE = 224
    NUM_CLASSES = 15
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ImageNet 标准化参数
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==================== 训练 ====================
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 80
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    FINETUNE_LR = 0.0001
    FINETUNE_EPOCHS = 30

    # ==================== 正则化 ====================
    DROPOUT_RATE = 0.5
    LABEL_SMOOTHING = 0.0

    # ==================== 早停 ====================
    EARLY_STOP_PATIENCE = 20

    # ==================== 混合精度 ====================
    USE_AMP = True

    # ==================== 梯度裁剪 ====================
    GRAD_CLIP_MAX_NORM = 1.0

    # ==================== 随机种子 ====================
    SEED = 42

    # ==================== 迁移学习 ====================
    FREEZE_BACKBONE = True
    BACKBONE = "resnet50"
