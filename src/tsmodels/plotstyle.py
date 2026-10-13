"""绘图统一样式：中文宋体五号，西文与数字 Times New Roman 五号。

约定：中文 10.5 磅（五号）宋体，英文与数字 10.5 磅 Times New Roman。
实现方式：matplotlib 逐字形字体回退（需 matplotlib >= 3.6），
西文字形命中 Times New Roman，缺失的中文字形回退到 SimSun。

用法：

    from tsmodels.plotstyle import apply_style, save_figure

    apply_style()
    fig, ax = plt.subplots(...)
    ...
    save_figure(fig, "docs/figures/fig_roc_a5d")   # 同时输出 PNG(300dpi) 与 SVG
"""

from pathlib import Path

import matplotlib

CJK_FONT = "SimSun"
WESTERN_FONT = "Times New Roman"
FONT_SIZE = 10.5  # 五号


def apply_style():
    """应用统一字体与画布样式；同时保留 SVG 文本（可编辑）。"""
    matplotlib.rcParams.update(
        {
            "font.family": [WESTERN_FONT, CJK_FONT],
            "font.serif": [WESTERN_FONT, CJK_FONT],
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
        }
    )


def save_figure(fig, stem):
    """将图同时保存为 PNG（300dpi，供文档排版）与 SVG（可编辑矢量源）。"""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path)
    fig.savefig(svg_path)
    return png_path, svg_path
