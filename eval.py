"""
模型评估脚本 —— 全面评估检测模型性能

用法: python eval.py

输出:
- mAP@50, mAP@75, mAP@50-95
- 每类 AP
- 混淆矩阵 (检测版)
- 检测结果可视化样例
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
import cv2
import numpy as np


def main():
    cfg = Config()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(" [ERROR] 请先安装 ultralytics: pip install ultralytics")
        sys.exit(1)

    # 加载最佳模型
    model_path = os.path.join(cfg.WEIGHT_DIR, 'best.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(cfg.OUTPUT_DIR, 'train', 'weights', 'best.pt')

    if not os.path.exists(model_path):
        print(f" [ERROR] 未找到模型权重: {model_path}")
        print(" 请先运行 train.py")
        return

    print(f"\n 加载模型: {model_path}")
    model = YOLO(model_path)

    yaml_path = os.path.join(cfg.YOLO_DATASET_DIR, 'dataset.yaml')

    # ========== 在验证集上评估 ==========
    print(f"\n 在验证集上评估...")
    results = model.val(
        data=yaml_path,
        split='val',
        imgsz=cfg.IMAGE_SIZE,
        batch=cfg.BATCH_SIZE,
        device=cfg.DEVICE,
        project=cfg.OUTPUT_DIR,
        name='eval_detailed',
        exist_ok=True,
        conf=cfg.CONF_THRESHOLD,
        iou=cfg.IOU_THRESHOLD,
    )

    # ========== 打印详细结果 ==========
    print("\n" + "=" * 65)
    print(" 评估结果")
    print("=" * 65)

    # 整体指标
    print(f"\n 整体指标:")
    print(f"   mAP@50:     {results.box.map50:.4f}")
    print(f"   mAP@75:     {results.box.map75:.4f}")
    print(f"   mAP@50-95:  {results.box.map:.4f}")

    # 每类指标
    print(f"\n 每类 AP (按 mAP@50 排序):")
    ap_per_class = {}
    if hasattr(results, 'boxes') and results.boxes is not None:
        ap50_list = results.box.ap50.flatten().tolist() if hasattr(results.box, 'ap50') else []
        for i, ap in enumerate(ap50_list):
            cls_name = cfg.CLASS_NAMES.get(i, f'class_{i}')
            cn_name = cfg.CLASS_CN.get(cls_name, cls_name)
            ap_per_class[cn_name] = ap
            bar = '█' * int(ap * 50) + '░' * (50 - int(ap * 50))
            print(f"   {cn_name:<8s}: AP@50={ap:.4f}  {bar}")

    # 找出最好和最差的类别
    if ap_per_class:
        best_cls = max(ap_per_class, key=ap_per_class.get)
        worst_cls = min(ap_per_class, key=ap_per_class.get)
        print(f"\n 精度最高: {best_cls} ({ap_per_class[best_cls]:.4f})")
        print(f" 精度最低: {worst_cls} ({ap_per_class[worst_cls]:.4f})")

    print(f"\n [完成] 评估结果保存在: {cfg.OUTPUT_DIR}/eval_detailed")


if __name__ == "__main__":
    main()
