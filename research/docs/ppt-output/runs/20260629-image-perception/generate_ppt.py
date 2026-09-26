"""Generate a 4-slide academic PPTX about image perception comparison experiments."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent / "output"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Colors
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x1A, 0x1A, 0x1A)
DARK = RGBColor(0x2D, 0x2D, 0x2D)
GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0xE8, 0xE8, 0xE8)
BLUE = RGBColor(0x2C, 0x5F, 0x8A)
LIGHT_BLUE = RGBColor(0xD6, 0xE8, 0xF7)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x27, 0xAE, 0x60)
ACCENT = RGBColor(0x3A, 0x7C, 0xBF)


def add_bg(slide, color=WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_title_bar(slide, title_text, subtitle_text=None):
    """Add a clean top title bar."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
        prs.slide_width, Inches(1.2)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = BLUE
    shape.line.fill.background()

    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(30)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.font.name = "Helvetica Neue"
    p.alignment = PP_ALIGN.LEFT
    tf.margin_left = Inches(0.8)
    tf.margin_top = Inches(0.15)

    if subtitle_text:
        shape2 = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.2),
            prs.slide_width, Inches(0.4)
        )
        shape2.fill.solid()
        shape2.fill.fore_color.rgb = LIGHT_BLUE
        shape2.line.fill.background()
        tf2 = shape2.text_frame
        tf2.margin_left = Inches(0.8)
        p2 = tf2.paragraphs[0]
        p2.text = subtitle_text
        p2.font.size = Pt(14)
        p2.font.color.rgb = BLUE
        p2.font.name = "Helvetica Neue"


def add_text_box(slide, left, top, width, height, text, font_size=16,
                 bold=False, color=DARK, alignment=PP_ALIGN.LEFT, font_name="Helvetica Neue"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return tf


def add_rich_text_box(slide, left, top, width, height, lines):
    """lines: list of (text, font_size, bold, color)"""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (text, font_size, bold, color) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = "Helvetica Neue"
        p.space_after = Pt(4)
    return tf


def add_card(slide, left, top, width, height, title, items, title_color=BLUE, bg_color=WHITE):
    """Add a card with title and bullet items."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = LIGHT_GRAY
    shape.line.width = Pt(1)

    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.3)
    tf.margin_right = Inches(0.3)
    tf.margin_top = Inches(0.2)

    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = title_color
    p.font.name = "Helvetica Neue"
    p.space_after = Pt(8)

    for item in items:
        p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(12)
        p.font.color.rgb = DARK
        p.font.name = "Helvetica Neue"
        p.space_after = Pt(2)
        p.level = 0


def add_flow_arrow(slide, left, top, width, height, text, color=BLUE):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()

    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(11)
    p.font.color.rgb = WHITE
    p.font.name = "Helvetica Neue"
    p.alignment = PP_ALIGN.CENTER


def add_page_number(slide, num):
    add_text_box(slide, 12.5, 7.0, 0.8, 0.4, str(num), 10, color=GRAY, alignment=PP_ALIGN.RIGHT)


# ================================================================
# SLIDE 1: Title / Cover
# ================================================================
slide1 = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_bg(slide1, WHITE)

# Large blue block
shape = slide1.shapes.add_shape(
    MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
    prs.slide_width, Inches(4.5)
)
shape.fill.solid()
shape.fill.fore_color.rgb = BLUE
shape.line.fill.background()

add_text_box(slide1, 1.5, 0.8, 10, 1.0,
             "图像感知相似度对比实验",
             36, bold=True, color=WHITE, font_name="Helvetica Neue")

add_text_box(slide1, 1.5, 1.8, 10, 0.6,
             "基于 ViT / CLIP 嵌入与感知哈希的图像变体评估",
             18, color=RGBColor(0xBB, 0xD5, 0xEF), font_name="Helvetica Neue")

# Subtitle block on blue
add_text_box(slide1, 1.5, 2.8, 10, 1.2,
             "Experiment 1: 多操作组合变体生成与评估  (793 张变体)\n"
             "Experiment 2: 单操作参数化变体生成与评估  (低/中/高三档)",
             16, color=RGBColor(0xDD, 0xEE, 0xF7), font_name="Helvetica Neue")

# Bottom section: overview cards
for i, (title, desc) in enumerate([
    ("12 种操作", "翻转 · 裁剪 · 平移\n旋转 · 亮度 · 暗角\n色温 · 锐化 · 倾斜\n胶片噪点"),
    ("6 项评估指标", "ViT Cosine  CLIP Cosine\npHash  aHash  dHash\nwHash  PDQ"),
    ("2 个实验范式", "Ex1: 组合生成 (1~4 ops)\nEx2: 参数化变体\n(low / mid / high)"),
]):
    left = 1.5 + i * 3.5
    add_card(slide1, left, 5.2, 3.2, 2.0, title, desc.split("\n"), title_color=BLUE)

add_page_number(slide1, 1)

# ================================================================
# SLIDE 2: Experiment 1 — Combination Generation
# ================================================================
slide2 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide2, WHITE)
add_title_bar(slide2, "Experiment 1: 多操作组合生成", "generate_ex1.py → 12 种基础操作的 C(12,1)+C(12,2)+C(12,3)+C(12,4) = 793 张变体")

# Operations grid
ops = [
    ("F1", "水平翻转"), ("F2", "垂直翻转"), ("C", "裁剪+缩放"),
    ("S1", "水平平移"), ("S2", "垂直平移"), ("R", "旋转 2°"),
    ("B", "亮度/对比度"), ("V", "暗角"), ("T", "色温偏移"),
    ("Sh", "锐化"), ("K", "透视倾斜"), ("N", "胶片噪点"),
]
for i, (sym, name) in enumerate(ops):
    col = i % 4
    row = i // 4
    left = 0.5 + col * 2.0
    top = 2.0 + row * 0.75

    shape = slide2.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(1.8), Inches(0.6)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT_BLUE if i % 2 == 0 else WHITE
    shape.line.color.rgb = LIGHT_GRAY
    shape.line.width = Pt(0.5)

    tf = shape.text_frame
    tf.margin_left = Inches(0.1)
    p = tf.paragraphs[0]
    p.text = f" {sym} {name}"
    p.font.size = Pt(11)
    p.font.color.rgb = DARK
    p.font.name = "Helvetica Neue"

# Combination math
add_rich_text_box(slide2, 0.5, 4.5, 8, 1.2, [
    ("生成公式", 16, True, BLUE),
    ("C(12,1)=12 + C(12,2)=66 + C(12,3)=220 + C(12,4)=495 = 793 张", 14, False, DARK),
])

# Flow
for j, (label, lx) in enumerate([("1-操作组合\n12 张", 0.5), ("2-操作组合\n66 张", 2.5), ("3-操作组合\n220 张", 4.5), ("4-操作组合\n495 张", 6.5)]):
    add_flow_arrow(slide2, lx, 5.6, 1.8, 0.7, label,
                   [BLUE, RGBColor(0x3A, 0x7C, 0xBF), RGBColor(0x5B, 0x9B, 0xD5), RED][j])

# Right side: evaluation
add_card(slide2, 8.8, 2.0, 4.0, 3.5, "评估方法", [
    "ViT (vit-base-patch16-224)",
    "  余弦相似度 (0~1)",
    "CLIP (clip-vit-base-patch32)",
    "  余弦相似度 (0~1)",
    "5 种感知哈希距离:",
    "  pHash / aHash / dHash",
    "  wHash / PDQ",
    "",
    "→ test_report.xlsx",
], title_color=BLUE)

add_page_number(slide2, 2)

# ================================================================
# SLIDE 3: Experiment 2 — Parameterized Variants
# ================================================================
slide3 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide3, WHITE)
add_title_bar(slide3, "Experiment 2: 单操作参数化变体", "generate_ex2.py → 每种操作 low / mid / high 三档强度，独立生成 ~30 张变体")

# Operation settings table
settings = [
    ("C", "裁剪缩放", "crop 2%", "crop 5%", "crop 8%"),
    ("S1/S2", "平移", "shift 2%", "shift 5%", "shift 8%"),
    ("R", "旋转", "3°", "6°", "12°"),
    ("B", "亮度/对比度", "α=1.02 β=4", "α=1.04 β=8", "α=1.06 β=12"),
    ("V", "暗角", "dark=0.10", "dark=0.25", "dark=0.40"),
    ("T", "色温", "R1.03 B0.97", "R1.06 B0.95", "R1.09 B0.93"),
    ("Sh", "锐化", "weight 0.20", "weight 0.40", "weight 0.60"),
    ("K", "透视倾斜", "margin 0.01", "margin 0.03", "margin 0.05"),
    ("N", "胶片噪点", "σ=1.5", "σ=3.0", "σ=4.5"),
]

# Table header (table: 5 cols × 1.5" = 7.5" wide, ends at ~8.1")
headers = ["操作", "名称", "Low", "Mid", "High"]
for i, h in enumerate(headers):
    add_text_box(slide3, 0.4 + i * 1.5, 1.9, 1.5, 0.4, h, 12, bold=True, color=WHITE)
    shape = slide3.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.4 + i * 1.5), Inches(1.9),
        Inches(1.5), Inches(0.35)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = BLUE
    shape.line.fill.background()
    add_text_box(slide3, 0.4 + i * 1.5, 1.92, 1.5, 0.35, h, 11, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

for r, (op, name, low, mid, high) in enumerate(settings):
    row_bg = LIGHT_BLUE if r % 2 == 0 else WHITE
    for c, val in enumerate([op, name, low, mid, high]):
        shape = slide3.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.4 + c * 1.5), Inches(2.3 + r * 0.45),
            Inches(1.5), Inches(0.45)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = row_bg
        shape.line.color.rgb = LIGHT_GRAY
        shape.line.width = Pt(0.3)
        add_text_box(slide3, 0.45 + c * 1.5, 2.33 + r * 0.45, 1.4, 0.4,
                     val, 10, color=DARK, alignment=PP_ALIGN.CENTER if c >= 2 else PP_ALIGN.LEFT)

# F1/F2 note
add_text_box(slide3, 0.4, 6.5, 8, 0.5,
             "* F1 (水平翻转) 和 F2 (垂直翻转) 为不可调参数操作，各仅生成 1 张",
             11, color=GRAY)

# Right side: evaluation summary (moved left to align with narrower table)
add_card(slide3, 8.3, 2.0, 4.5, 3.0, "评估流程", [
    "1. 对每张原图分别评估",
    "2. ViT + CLIP 余弦相似度",
    "3. 5 种感知哈希距离",
    "4. 每张原图 → 独立报告",
    "",
    "F1/F2 变体各 1 张",
    "其他操作各 3 张 = 30 张",
    "→ horses_report.xlsx",
], title_color=BLUE)

# Comparison box: Ex1 vs Ex2
add_card(slide3, 8.3, 5.3, 4.5, 1.7, "Ex1 vs Ex2 关键区别", [
    "Ex1: 操作组合，793 张",
    "Ex2: 单操作参数化，~30 张",
    "Ex1: 一张总报告",
    "Ex2: 每原图独立报告",
], title_color=RED)

add_page_number(slide3, 3)

# ================================================================
# SLIDE 4: Evaluation Framework + Summary
# ================================================================
slide4 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide4, WHITE)
add_title_bar(slide4, "评估框架与总结", "auto_perception_ex1.py / auto_perception_ex2.py — 深度学习嵌入 + 传统感知哈希 双重评估")

# Left: evaluation pipeline flow
add_text_box(slide4, 0.8, 2.0, 5, 0.4, "评估流程", 18, bold=True, color=BLUE)

flow_steps = [
    ("1. 加载模型", "ViT + CLIP 预训练模型\n感知哈希初始化"),
    ("2. 嵌入提取", "对每张图像计算\nViT/CLIP 归一化嵌入向量"),
    ("3. 哈希计算", "对每张图像计算\n5 种感知哈希值"),
    ("4. 距离/相似度", "余弦相似度 (ViT/CLIP)\n汉明距离 (哈希)"),
    ("5. 输出报告", "Excel 含缩略图\n+ 各项指标分数"),
]

for i, (title, desc) in enumerate(flow_steps):
    top = 2.5 + i * 0.95
    # Number circle
    shape = slide4.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(0.9), Inches(top + 0.05),
        Inches(0.35), Inches(0.35)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = BLUE
    shape.line.fill.background()
    tf = shape.text_frame
    p = tf.paragraphs[0]
    p.text = str(i + 1)
    p.font.size = Pt(11)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    add_rich_text_box(slide4, 1.5, top, 4.5, 0.85, [
        (title, 13, True, DARK),
        (desc, 11, False, GRAY),
    ])

    if i < 4:
        shape = slide4.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(1.05), Inches(top + 0.42),
            Inches(0.05), Inches(0.5)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_GRAY
        shape.line.fill.background()

# Right: Metrics comparison table
add_text_box(slide4, 7.0, 2.0, 5, 0.4, "评估指标体系", 18, bold=True, color=BLUE)

metrics = [
    ("类别", "指标", "范围", "含义"),
    ("深度学习", "ViT Cosine", "0 ~ 1", "越高越相似"),
    ("深度学习", "CLIP Cosine", "0 ~ 1", "越高越相似"),
    ("感知哈希", "pHash Dist", "0 ~ 1", "越低越相似"),
    ("感知哈希", "aHash Dist", "0 ~ 1", "越低越相似"),
    ("感知哈希", "dHash Dist", "0 ~ 1", "越低越相似"),
    ("感知哈希", "wHash Dist", "0 ~ 1", "越低越相似"),
    ("感知哈希", "PDQ Dist", "0 ~ 1", "越低越相似"),
]

for r, row_data in enumerate(metrics):
    for c, val in enumerate(row_data):
        is_header = (r == 0)
        bg = BLUE if is_header else (LIGHT_BLUE if r % 2 == 1 else WHITE)
        fg = WHITE if is_header else DARK
        shape = slide4.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(7.0 + c * 1.45), Inches(2.5 + r * 0.45),
            Inches(1.45), Inches(0.45)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg
        shape.line.color.rgb = LIGHT_GRAY
        shape.line.width = Pt(0.3)
        add_text_box(slide4, 7.1 + c * 1.45, 2.52 + r * 0.45, 1.3, 0.4,
                     val, 11, bold=is_header, color=fg, alignment=PP_ALIGN.CENTER)

# Bottom summary
add_rich_text_box(slide4, 0.8, 6.5, 12, 0.8, [
    ("核心结论", 16, True, BLUE),
    ("两个实验从不同维度评估图像操作的感知影响：Ex1 探索操作叠加的复合效应 (793 张)，Ex2 研究单一操作在不同强度下的独立影响 (32 张)。"
     "ViT/CLIP 深度嵌入与 5 种传统感知哈希形成互补的评估体系。", 12, False, GRAY),
])

add_page_number(slide4, 4)

# ================================================================
# SAVE
# ================================================================
output_file = OUTPUT_PATH / "image-perception-experiments.pptx"
prs.save(str(output_file))
print(f"PPTX saved to: {output_file}")
print(f"Slides: {len(prs.slides)}")

