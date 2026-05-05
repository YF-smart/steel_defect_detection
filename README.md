# 🔧 钢材表面缺陷检测系统

基于 **YOLOv8 + PyTorch** 的钢材表面缺陷目标检测系统，支持 **6 类常见钢材缺陷** 的自动检测与定位。

## 项目简介

本项目实现了从数据预处理（XML→YOLO格式转换、灰度图→RGB）、模型训练、评估到推理的完整目标检测 pipeline。数据集来自 NEU-DET（东北大学钢材缺陷数据集），适用于工业品控场景。

### 核心能力

- **数据工程**：VOC XML → YOLO txt 格式转换、灰度图三通道处理
- **目标检测**：YOLOv8 训练、mAP 评估、检测框可视化
- **工业场景**：小数据集增强策略（Mosaic / MixUp）、缺陷定位与分类

## 项目结构

```
steel_defect_detection/
├── config/config.py              # 全局配置
├── data/
│   └── convert_xml2yolo.py       # XML→YOLO转换 + 数据分析
├── utils/
│   └── plot_utils.py             # 可视化工具
├── train.py                      # 训练脚本
├── eval.py                       # 评估脚本
├── inference.py                  # 推理脚本（单张图片）
├── demo/                         # 测试样例
├── weights/README.md             # 权重说明
├── outputs/                      # 输出（模型、图表）
└── README.md
```

## 快速开始

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 准备数据
数据集路径: `E:\project_data\NEU-DET\NEU-DET\`

训练时会自动完成 XML→YOLO 转换。

### 3. 训练
```bash
python train.py
```

### 4. 评估
```bash
python eval.py
```

### 5. 推理
```bash
python inference.py demo/crazing_sample.jpg --save
```

## 数据集

| 缺陷类型 | 英文名 | 描述 |
|----------|--------|------|
| 裂纹 | crazing | 网状微裂纹 |
| 夹杂 | inclusion | 非金属夹杂物 |
| 斑块 | patches | 表面氧化斑 |
| 麻点 | pitted_surface | 点状凹坑 |
| 氧化皮 | rolled-in_scale | 轧制嵌入氧化铁皮 |
| 划痕 | scratches | 机械划伤 |

- 图片: 1800 张 (训练 1440 / 验证 360)
- 分辨率: 200×200 灰度图
- 每类约 300 张，类别较均衡

## 技术栈

| 技术 | 用途 |
|------|------|
| ultralytics (YOLOv8) | 目标检测框架 |
| PyTorch | 底层深度学习框架 |
| OpenCV | 图像读取、灰度转RGB |
| XML ElementTree | VOC 标注解析 |
| Matplotlib / Seaborn | 可视化 |

## 面试要点

1. **XML→YOLO 格式转换**：处理了 VOC 绝对坐标 → YOLO 归一化坐标的转换逻辑
2. **灰度图处理**：复制为三通道保留纹理信息，兼容 ImageNet 预训练权重
3. **小数据集策略**：1800 张图 + Mosaic/MixUp 增强防过拟合
4. **mAP 评估**：理解 mAP@50 vs mAP@50-95 的含义，能分析每类精度差异

## License

MIT
