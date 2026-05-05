"""
全局配置文件 —— 所有超参数和路径集中管理

项目: 钢材表面缺陷检测系统
任务: 目标检测 (YOLOv8)
数据集: NEU-DET (1800张灰度图，6类缺陷)
"""
import os


class Config:
    # ==================== 路径配置 ====================
    # 原始数据集路径
    DATASET_ROOT = r"E:\project_data\NEU-DET\NEU-DET"

    # 项目路径
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
    WEIGHT_DIR = os.path.join(PROJECT_DIR, "weights")
    DEMO_DIR = os.path.join(PROJECT_DIR, "demo")

    # 转换后的 YOLO 格式数据集路径
    YOLO_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset")

    # ==================== 模型配置 ====================
    MODEL_NAME = "yolov8n"  # n=nano (最轻量), s=small, m=medium, l=large
    # 选 yolov8n 的原因：数据量小(1800张)、缺陷特征相对简单，
    # nano 模型只有 3.2M 参数，训练快、推理快、够用

    IMAGE_SIZE = 640        # YOLO 标准输入尺寸，图片会被自动缩放
    NUM_CLASSES = 6

    # ==================== 训练配置 ====================
    EPOCHS = 100            # YOLOv8 默认 100 轮，数据量小可适当减少
    BATCH_SIZE = 16         # 目标检测比分类吃内存，设小一点
    LR = 0.01               # YOLOv8 默认学习率
    DEVICE = "cuda"         # "cuda" 或 "cpu"
    NUM_WORKERS = 2

    # ==================== 缺陷类别 ====================
    # NEU-DET 的 6 类钢材表面缺陷
    CLASS_NAMES = {
        0: "crazing",           # 裂纹 (网状微裂纹)
        1: "inclusion",         # 夹杂 (非金属夹杂物)
        2: "patches",           # 斑块 (表面氧化斑)
        3: "pitted_surface",    # 麻点 (点状凹坑)
        4: "rolled-in_scale",   # 氧化皮 (轧制嵌入的氧化铁皮)
        5: "scratches",         # 划痕 (机械划伤)
    }

    CLASS_CN = {
        "crazing": "裂纹",
        "inclusion": "夹杂",
        "patches": "斑块",
        "pitted_surface": "麻点",
        "rolled-in_scale": "氧化皮",
        "scratches": "划痕",
    }

    # ==================== 评估配置 ====================
    # 目标检测的核心指标是 mAP (mean Average Precision)
    CONF_THRESHOLD = 0.25   # 置信度阈值，低于此值的检测框被过滤
    IOU_THRESHOLD = 0.45    # NMS 的 IoU 阈值


def get_config():
    return Config()
