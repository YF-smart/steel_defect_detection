"""
训练主脚本 —— YOLOv8 钢材缺陷检测

用法: python train.py

流程:
1. 数据转换 (XML → YOLO)
2. 数据分析
3. YOLOv8 训练
4. 模型评估
5. 生成可视化结果

面试可讲:
- 为什么选 YOLOv8: 工业界最主流的目标检测框架, 从训练到部署一条龙
- 为什么选 nano: 数据量小(1800张), 模型太大容易过拟合
- 灰度图的处理: 复制成三通道保留纹理信息, 同时兼容预训练权重
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from data.convert_xml2yolo import convert_dataset, analyze_dataset


def main():
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ==================== 1. 数据分析 ====================
    print("\n[1/4] 分析原始数据集...")
    analyze_dataset(cfg)

    # ==================== 2. 数据转换 (XML → YOLO) ====================
    print("\n[2/4] 转换数据格式 (XML → YOLO)...")
    yaml_path = convert_dataset(cfg)

    # ==================== 3. 训练 ====================
    print("\n[3/4] 开始训练...")
    print("-" * 65)

    # 检查 ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print(" [ERROR] 请先安装 ultralytics: pip install ultralytics")
        sys.exit(1)

    # 创建模型 (加载预训练权重)
    model = YOLO(f"{cfg.MODEL_NAME}.pt")  # 自动下载预训练权重

    # 训练参数
    train_results = model.train(
        data=yaml_path,
        epochs=cfg.EPOCHS,
        imgsz=cfg.IMAGE_SIZE,
        batch=cfg.BATCH_SIZE,
        lr0=cfg.LR,
        device=cfg.DEVICE,
        workers=cfg.NUM_WORKERS,
        # 数据增强: YOLOv8 默认自带 Mosaic, MixUp, HSV 增强等
        # 小数据集保留 mosaic, 增强泛化
        mosaic=1.0,       # Mosaic 增强 (4张拼接成1张)
        mixup=0.1,        # MixUp 增强
        hsv_h=0.015,      # HSV 色调抖动
        hsv_s=0.7,        # HSV 饱和度抖动
        hsv_v=0.4,        # HSV 亮度抖动
        # 保存设置
        project=cfg.OUTPUT_DIR,
        name='train',
        exist_ok=True,
        save=True,
        save_period=10,   # 每10轮保存一次
        # 早停 & 验证
        patience=20,      # 早停容忍度
        val=True,
        # 预训练
        pretrained=True,
    )

    # ==================== 4. 生成评估结果 ====================
    print("\n[4/4] 生成评估结果...")

    # 复制最佳模型
    best_src = os.path.join(cfg.OUTPUT_DIR, 'train', 'weights', 'best.pt')
    if os.path.exists(best_src):
        best_dst = os.path.join(cfg.WEIGHT_DIR, 'best.pt')
        shutil.copy(best_src, best_dst)
        print(f" [OK] 最佳模型已保存: {best_dst}")

    # 在验证集上评估
    val_results = model.val(
        data=yaml_path,
        split='val',
        imgsz=cfg.IMAGE_SIZE,
        batch=cfg.BATCH_SIZE,
        device=cfg.DEVICE,
        project=cfg.OUTPUT_DIR,
        name='eval',
        exist_ok=True,
    )

    # 打印结果摘要
    print("\n" + "=" * 65)
    print(" 训练完成! 结果摘要")
    print("=" * 65)
    print(f" 最佳 mAP@50:     {train_results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f" 最佳 mAP@50-95:  {train_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f" 模型保存在:      {cfg.WEIGHT_DIR}")
    print(f" 训练输出在:      {cfg.OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
