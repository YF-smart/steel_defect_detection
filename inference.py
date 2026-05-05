"""
推理脚本 —— 输入钢材表面图片，输出缺陷检测结果

用法:
    python inference.py <图片路径>
    python inference.py demo/sample.jpg --save

功能:
    1. 加载 YOLOv8 训练好的模型
    2. 对输入图片进行检测
    3. 绘制检测框和标签
    4. 输出检测到的缺陷列表
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
import cv2


def main():
    parser = argparse.ArgumentParser(description='钢材表面缺陷检测')
    parser.add_argument('image', nargs='?', default=None,
                        help='输入图片路径')
    parser.add_argument('--save', action='store_true',
                        help='保存标注了检测结果的图片')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值 (默认0.25)')
    args = parser.parse_args()

    cfg = Config()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(" [ERROR] 请先安装 ultralytics: pip install ultralytics")
        sys.exit(1)

    # ========== 加载模型 ==========
    model_path = os.path.join(cfg.WEIGHT_DIR, 'best.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(cfg.OUTPUT_DIR, 'train', 'weights', 'best.pt')

    if not os.path.exists(model_path):
        print(f" [ERROR] 未找到模型权重!")
        print(f" 检查: {model_path}")
        print(" 请先运行 train.py")
        return

    model = YOLO(model_path)
    print(f" [OK] 模型已加载: {cfg.MODEL_NAME}")
    print(f" [OK] 类别数: {cfg.NUM_CLASSES}")

    # ========== 获取图片路径 ==========
    if args.image:
        image_path = args.image
    else:
        # 交互模式
        print("\n" + "=" * 60)
        print(" 钢材缺陷检测 (输入 'q' 退出)")
        print("=" * 60)
        image_path = input("\n 请输入图片路径: ").strip().strip('"')
        if image_path.lower() == 'q':
            return

    if not os.path.exists(image_path):
        print(f" [ERROR] 文件不存在: {image_path}")
        return

    # ========== 推理 ==========
    print(f"\n 检测: {image_path}")
    results = model.predict(
        source=image_path,
        conf=args.conf,
        device=cfg.DEVICE,
        verbose=False,
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("\n 未检测到任何缺陷")
        return

    # 打印检测结果
    print(f"\n 检测到 {len(result.boxes)} 处缺陷:")
    print(f" {'类别':<12s} {'置信度':>8s}  {'坐标 (x1,y1,x2,y2)'}")
    print("-" * 55)

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        cls_name = cfg.CLASS_NAMES.get(cls_id, f'cls_{cls_id}')
        cn_name = cfg.CLASS_CN.get(cls_name, cls_name)

        print(f" {cn_name:<12s} {conf:>7.1%}  "
              f"({xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f})")

    # ========== 保存结果图 ==========
    if args.save:
        save_dir = os.path.join(cfg.OUTPUT_DIR, 'inference')
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_detected.jpg")

        # YOLOv8 自带绘图功能
        annotated = result.plot()
        cv2.imwrite(save_path, annotated)
        print(f"\n [OK] 检测结果已保存: {save_path}")

        # 显示原图 vs 检测结果 (需要 PIL)
        try:
            from PIL import Image
            original = Image.open(image_path).convert('RGB')
            detected = Image.open(save_path)

            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            ax1.imshow(original, cmap='gray')
            ax1.set_title('原始图片', fontsize=13)
            ax1.axis('off')
            ax2.imshow(detected)
            ax2.set_title(f'检测结果 (共{len(result.boxes)}处缺陷)', fontsize=13)
            ax2.axis('off')
            plt.tight_layout()

            compare_path = os.path.join(save_dir, f"{base_name}_compare.jpg")
            plt.savefig(compare_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f" [OK] 对比图已保存: {compare_path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
