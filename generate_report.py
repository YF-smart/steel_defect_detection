"""生成项目知识讲解 PDF"""
import os, sys

try:
    from fpdf import FPDF
except ImportError:
    print("[ERROR] pip install fpdf2"); sys.exit(1)

import matplotlib.font_manager as fm

def find_chinese_font():
    for d in ["C:/Windows/Fonts/", "/usr/share/fonts/"]:
        for f in ["msyh.ttc", "msyh.ttf", "simhei.ttf", "simsun.ttc"]:
            p = os.path.join(d, f)
            if os.path.exists(p): return p
    return None

class PDF(FPDF):
    def __init__(self, fp):
        super().__init__()
        self.add_font('CJK', '', fp)
        self.add_font('CJK', 'B', fp)
    def header(self):
        if self.page_no() == 1: return
        self.set_font('CJK', '', 9); self.set_text_color(128,128,128)
        self.cell(0, 8, '钢材表面缺陷检测系统 — 知识详解', align='C'); self.ln(12)
    def footer(self):
        if self.page_no() == 1: return
        self.set_y(-15); self.set_font('CJK', '', 8); self.set_text_color(128,128,128)
        self.cell(0, 10, f'- {self.page_no()} -', align='C')
    def title_page(self):
        self.add_page(); self.ln(50)
        self.set_font('CJK', 'B', 28); self.set_text_color(33, 150, 243)
        self.multi_cell(0, 15, '钢材表面缺陷\n检测系统', align='C'); self.ln(10)
        self.set_font('CJK', '', 16); self.set_text_color(100,100,100)
        self.cell(0, 12, '项目知识详解与面试指南', align='C'); self.ln(30)
        self.set_font('CJK', '', 12); self.set_text_color(80,80,80)
        self.cell(0, 10, '技术栈: YOLOv8 + PyTorch + OpenCV', align='C'); self.ln(10)
        self.cell(0, 10, '任务类型: 目标检测 / 工业缺陷检测', align='C'); self.ln(10)
        self.cell(0, 10, '数据集: NEU-DET (1800张/6类钢材缺陷)', align='C'); self.ln(40)
        self.set_font('CJK', '', 10); self.set_text_color(150,150,150)
        self.cell(0, 10, '2026年5月', align='C')
    def s1(self, t):
        self.ln(5); self.set_font('CJK', 'B', 16); self.set_text_color(33,150,243)
        self.cell(0, 12, t); self.ln(14)
        self.set_draw_color(33,150,243); self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w-self.r_margin, self.get_y()); self.ln(5)
    def s2(self, t):
        self.ln(3); self.set_font('CJK', 'B', 13); self.set_text_color(60,60,60)
        self.cell(0, 10, t); self.ln(12)
    def body(self, t):
        self.set_font('CJK', '', 10); self.set_text_color(50,50,50)
        self.multi_cell(0, 6.5, t); self.ln(1)
    def bullet(self, t):
        self.set_font('CJK', '', 10); self.set_text_color(50,50,50)
        self.set_x(self.l_margin + 5); self.cell(5, 6, '-')
        self.multi_cell(self.w-self.r_margin-self.l_margin-10, 6.5, t); self.ln(0.5)
    def hl(self, t):
        self.ln(2); self.set_fill_color(232, 245, 253); self.set_draw_color(33,150,243)
        self.set_font('CJK', 'B', 10); self.set_text_color(0,100,200)
        self.set_x(self.l_margin+3)
        self.multi_cell(self.w-self.r_margin-self.l_margin-6, 6.5, f'> {t}', fill=True); self.ln(3)
    def table(self, headers, rows):
        cw = self.w / len(headers)
        self.set_font('CJK', 'B', 9); self.set_fill_color(33,150,243); self.set_text_color(255,255,255)
        for h in headers: self.cell(cw, 8, h, border=1, fill=True, align='C')
        self.ln()
        self.set_font('CJK', '', 9)
        for ri, r in enumerate(rows):
            self.set_fill_color(240,248,255) if ri%2==0 else self.set_fill_color(255,255,255)
            self.set_text_color(50,50,50)
            for c in r: self.cell(cw, 7, str(c), border=1, fill=True, align='C')
            self.ln()
        self.ln(3)

def generate():
    fp = find_chinese_font()
    if not fp: print("[ERROR] No Chinese font found"); return
    print(f"[OK] Font: {fp}")

    out = os.path.join(os.path.dirname(__file__), 'outputs', '钢材缺陷检测系统_知识详解.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    pdf = PDF(fp); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.title_page()

    # 一
    pdf.s1('一、项目概述')
    pdf.body('基于 YOLOv8 的钢材表面缺陷目标检测系统，能检测 6 类常见钢材缺陷（裂纹、夹杂、斑块、麻点、氧化皮、划痕），输出每个缺陷的类别、位置和置信度。对标工业品控 / AI 质检岗位。')
    pdf.hl('面试定位：展示数据工程能力（XML->YOLO转换、灰度图处理）和 YOLOv8 目标检测全流程（训练、mAP评估、推理）。')

    # 二
    pdf.s1('二、数据集 NEU-DET')
    pdf.s2('2.1 数据概况')
    pdf.body('东北大学发布的钢材表面缺陷数据集。1800 张灰度图（200x200），6 类缺陷，类别较均衡。原始标注为 VOC XML 格式。')
    pdf.table(['缺陷', '英文', '特征', 'AP@50'],
        [['裂纹', 'crazing', '网状微裂纹', '0.494'],
         ['夹杂', 'inclusion', '深色不规则块', '0.796'],
         ['斑块', 'patches', '表面氧化斑', '0.930'],
         ['麻点', 'pitted_surface', '点状凹坑', '0.820'],
         ['氧化皮', 'rolled-in_scale', '氧化铁皮轧入', '0.565'],
         ['划痕', 'scratches', '机械划伤', '0.835']])

    pdf.s2('2.2 数据特殊挑战')
    pdf.bullet('灰度图: 预训练模型在 RGB 上训练，需转三通道以兼容预训练权重')
    pdf.bullet('标注格式: VOC XML 绝对坐标 -> YOLO 归一化坐标转换')
    pdf.bullet('数据量小: 1800 张需配合 Mosaic/MixUp 增强防过拟合')
    pdf.bullet('裂纹极细: 200x200 图上仅几像素宽，是业界公认最难检测的缺陷之一')

    # 三
    pdf.s1('三、核心技术知识')
    pdf.s2('3.1 XML -> YOLO 格式转换')
    pdf.body('VOC格式存储绝对像素坐标 (xmin,ymin,xmax,ymax)，YOLO需要归一化中心坐标 (cx,cy,w,h)。转换公式: cx=(xmin+xmax)/2/W, cy=(ymin+ymax)/2/H, w=(xmax-xmin)/W, h=(ymax-ymin)/H。归一化后坐标与图片尺寸无关，模型可处理任意分辨率输入。')
    pdf.s2('3.2 灰度图 -> RGB')
    pdf.body('用 PIL Image.convert("RGB") 将单通道复制为三通道，保留所有纹理信息，同时兼容 ImageNet 预训练权重。')
    pdf.s2('3.3 mAP 评估指标')
    pdf.body('目标检测不用 Accuracy，用 mAP。mAP@50 表示 IoU>=0.5 算命中，工业界最常用；mAP@50-95 更严格(要求定位更准)。本项目 mAP@50=0.74，斑块类最高(0.93)，裂纹类最低(0.49)。')
    pdf.s2('3.4 YOLOv8 检测原理')
    pdf.body('将图片分 SxS 网格，每个网格预测多个边界框+置信度+类别。一次前向传播同时完成定位和分类，比两阶段检测器(RCNN系列)快一个数量级。YOLOv8n 仅 3M 参数，6.3MB，推理仅需 2ms/张。')

    # 四
    pdf.s1('四、代码架构')
    pdf.bullet('config/config.py - 全局配置(路径、类别名、超参数)')
    pdf.bullet('data/convert_xml2yolo.py - XML->YOLO转换 + 灰度转RGB + 数据分析')
    pdf.bullet('utils/plot_utils.py - 可视化: 训练曲线、检测框、混淆矩阵')
    pdf.bullet('train.py - 一键训练: 自动转换数据 -> YOLOv8训练 -> 评估')
    pdf.bullet('eval.py - 模型评估: mAP、每类AP、易混淆类别对')
    pdf.bullet('inference.py - 单图推理: 加载模型 -> 检测 -> 绘制结果')

    # 五
    pdf.s1('五、面试话术')
    pdf.s2('自我介绍')
    pdf.hl('"我做了个钢材表面缺陷检测系统，基于 YOLOv8，能检测 6 类缺陷。项目涵盖从 VOC XML 标注转 YOLO 格式、灰度图处理，到训练、mAP 评估和推理的完整流程。最终 mAP@50 达到 0.74。"')
    pdf.s2('被问"负责什么模块"')
    pdf.bullet('数据工程: 处理 VOC XML 格式转 YOLO，灰度图转 RGB 三通道适配预训练权重')
    pdf.bullet('模型训练: YOLOv8n + Mosaic/MixUp 增强，调优早停和学习率')
    pdf.bullet('评估分析: 分析每类 AP，发现裂纹(0.49)和氧化皮(0.56)是难点，能解释原因')
    pdf.s2('被问"裂纹为什么精度低"')
    pdf.body('裂纹在 200x200 的图上只有几个像素宽，属于极小目标检测。YOLOv8n 的感受野相对较大，对小目标不友好。改进方向: 提高输入分辨率、用更大的模型(YOLOv8s)、或专门增加裂纹类别的数据增强。')
    pdf.s2('被问"和项目一(分类)的区别"')
    pdf.body('项目一是图像分类(判断是什么)，输出一个类别标签。项目二是目标检测(判断是什么+在哪里)，输出多个(类别+坐标+置信度)。检测比分类多一个定位任务，评估指标也从 Accuracy 变成 mAP。')

    # 六
    pdf.s1('六、最终结果')
    pdf.table(['指标', '数值'],
        [['mAP@50', '0.740'], ['mAP@50-95', '0.391'], ['模型', 'YOLOv8n (3M)'],
         ['大小', '6.3 MB'], ['推理速度', '2.1ms/张'],
         ['训练耗时', '0.4小时 (RTX 3060)']])

    pdf.output(out)
    print(f"[DONE] PDF: {out}  ({pdf.page_no()} pages)")

if __name__ == "__main__":
    generate()
