"""
可视化工具 —— 训练曲线、混淆矩阵、检测结果可视化
"""
import os
import numpy as np
import matplotlib
import matplotlib.font_manager
matplotlib.font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_training_curves(results_dict, save_path):
    """
    绘制 YOLOv8 训练过程中的 Loss 和 mAP 曲线

    YOLOv8 训练返回的 results_dict 包含:
    - train/box_loss, train/cls_loss: 训练损失
    - val/box_loss, val/cls_loss: 验证损失
    - metrics/mAP50(B), metrics/mAP50-95(B): 验证 mAP
    """
    epochs = range(1, len(results_dict.get('train/box_loss', [])) + 1)
    if not epochs:
        print("[警告] 无训练数据可绘制")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上: Box Loss (边框回归损失)
    ax = axes[0, 0]
    ax.plot(epochs, results_dict.get('train/box_loss', []), 'b-o', label='Train', markersize=2)
    ax.plot(epochs, results_dict.get('val/box_loss', []), 'r-o', label='Val', markersize=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Bounding Box Loss (边框回归损失)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 右上: Classification Loss (分类损失)
    ax = axes[0, 1]
    ax.plot(epochs, results_dict.get('train/cls_loss', []), 'b-o', label='Train', markersize=2)
    ax.plot(epochs, results_dict.get('val/cls_loss', []), 'r-o', label='Val', markersize=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Classification Loss (分类损失)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 左下: mAP@50 (IoU=0.5 时的平均精度)
    ax = axes[1, 0]
    ax.plot(epochs, results_dict.get('metrics/mAP50(B)', []), 'g-o', markersize=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('mAP@50')
    ax.set_title('mAP@50 (IoU阈值=0.5)')
    ax.grid(True, alpha=0.3)

    # 右下: mAP@50-95 (IoU 从 0.5 到 0.95 的平均)
    ax = axes[1, 1]
    ax.plot(epochs, results_dict.get('metrics/mAP50-95(B)', []), 'm-o', markersize=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('mAP@50-95')
    ax.set_title('mAP@50-95 (IoU阈值=0.5~0.95)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" [OK] 训练曲线已保存: {save_path}")
    plt.close()


def plot_detection_results(image_path, detections, class_names, save_path):
    """
    在图片上绘制检测框和标签

    Args:
        image_path: 原始图片路径
        detections: YOLO 检测结果 [(x1,y1,x2,y2,conf,cls_id), ...]
        class_names: 类别名称字典
        save_path: 保存路径
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    colors = ['#E53935', '#FF9800', '#FDD835', '#4CAF50', '#2196F3', '#9C27B0']

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        color = colors[int(cls_id) % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        label = f"{class_names.get(int(cls_id), f'cls_{int(cls_id)}')} {conf:.2f}"
        ax.text(x1, y1 - 5, label, fontsize=9,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
                color='white')

    ax.axis('off')
    ax.set_title('缺陷检测结果', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" [OK] 检测结果图已保存: {save_path}")
    plt.close()


def plot_confusion_matrix(cm_data, class_names, save_path):
    """
    绘制混淆矩阵 (目标检测版)

    YOLOv8 输出的混淆矩阵已经是 numpy array，
    直接热力图展示即可
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_data, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=0, vmax=1.0)
    ax.set_xlabel('预测类别', fontsize=12)
    ax.set_ylabel('真实类别', fontsize=12)
    ax.set_title('混淆矩阵 (归一化)', fontsize=14)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_class_ap(ap_dict, save_path):
    """绘制每类的 AP 柱状图"""
    names = list(ap_dict.keys())
    values = list(ap_dict.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#E53935', '#FF9800', '#FDD835', '#4CAF50', '#2196F3', '#9C27B0']
    bars = ax.bar(names, values, color=colors[:len(names)])

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('Average Precision (AP)', fontsize=12)
    ax.set_title('每类 AP (Average Precision)', fontsize=14)
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
