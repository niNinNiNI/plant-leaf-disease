#!/bin/bash
# ============================================================
# 打包项目代码，准备上传到 Kaggle
# ============================================================
# 用法:
#   chmod +x prepare_for_kaggle.sh
#   ./prepare_for_kaggle.sh
#
# 生成两个文件:
#   1. plant_code.tar.gz  → 上传为 Kaggle Dataset（代码）
#   2. plantvillage.tar.gz → 上传为 Kaggle Dataset（数据集）
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "打包项目代码 → plant_code.tar.gz"
echo "============================================"

tar -czf plant_code.tar.gz \
    --exclude='checkpoints' \
    --exclude='results' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='archive' \
    --exclude='*.tar.gz' \
    --exclude='*.pth' \
    --exclude='.claude' \
    main.py requirements.txt src/

echo "✅ plant_code.tar.gz 已生成 ($(du -h plant_code.tar.gz | cut -f1))"

# ============================================================
# 数据集打包（如果数据在本地）
# ============================================================
DATA_DIR=""
for candidate in \
    "../archive/plantvillage/PlantVillage" \
    "archive/plantvillage/PlantVillage" \
    "../小组作业/archive/plantvillage/PlantVillage"; do
    if [ -d "$candidate" ]; then
        DATA_DIR="$candidate"
        break
    fi
done

if [ -n "$DATA_DIR" ]; then
    echo ""
    echo "============================================"
    echo "打包数据集 → plantvillage.tar.gz"
    echo "============================================"
    echo "数据目录: $DATA_DIR"
    echo "注意: 数据集较大，请耐心等待..."
    echo ""

    tar -czf plantvillage.tar.gz -C "$(dirname "$DATA_DIR")" "$(basename "$DATA_DIR")"

    echo "✅ plantvillage.tar.gz 已生成 ($(du -h plantvillage.tar.gz | cut -f1))"
else
    echo ""
    echo "⚠️  未找到数据集目录，请手动打包或通过 Kaggle 网页上传数据集"
fi

echo ""
echo "============================================"
echo "下一步操作:"
echo "============================================"
echo ""
echo "1. 打开 https://www.kaggle.com"
echo "2. Create → New Dataset → 上传 plant_code.tar.gz"
echo "   命名为: plant-disease-code"
echo "3. Create → New Dataset → 上传 plantvillage.tar.gz"
echo "   命名为: plantvillage-dataset"
echo "4. Create → New Notebook → 复制 kaggle_notebook.ipynb 的内容"
echo "5. 在 Notebook 左侧 Add Data → 添加以上两个数据集"
echo "6. Run All → 等待训练完成"
echo ""
