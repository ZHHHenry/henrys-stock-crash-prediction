"""生成 v1.1 多模型对比图表（结果图 + 结构图）。

输入：outputs/models/<模型>/<代际>/{metrics_*.json, scores_*.npz, panel_stats.json}
输出：docs/figures/*.png（300dpi，供文档排版）与 *.svg（可编辑矢量源）

用法（仓库根目录）：

    python scripts\\make_figures.py

字体：中文宋体五号（10.5pt），英文与数字 Times New Roman 10.5pt（见 tsmodels/plotstyle.py）。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsmodels.plotstyle import apply_style, save_figure

MODELS = ["lstm", "gru", "tcn", "lgbm", "logreg"]
DISPLAY = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "lgbm": "LightGBM", "logreg": "Logistic"}
COLORS = {
    "lstm": "#1f77b4",
    "gru": "#ff7f0e",
    "tcn": "#2ca02c",
    "lgbm": "#d62728",
    "logreg": "#9467bd",
}
LABELS = {"labelA5d": "A5d（未来 5 日单日收益 < −8%）", "labelB4w": "B4w（未来 4 周崩盘周）"}
MODELS_DIR = ROOT / "outputs" / "models"
FIG_DIR = ROOT / "docs" / "figures"


def metrics_path(model, gen, label, tag):
    return MODELS_DIR / model / gen / f"metrics_{label}_panel{tag}.json"


def scores_path(model, gen, label, tag):
    return MODELS_DIR / model / gen / f"scores_{label}_panel{tag}.npz"


def load_metrics(model, gen="gen5", label="labelA5d", tag=""):
    with open(metrics_path(model, gen, label, tag), encoding="utf-8") as f:
        return json.load(f)["metrics"]


def load_scores(model, gen="gen5", label="labelA5d", tag=""):
    data = np.load(scores_path(model, gen, label, tag))
    return data["y_true"], data["y_prob"]


def save(fig, name):
    png, svg = save_figure(fig, FIG_DIR / name)
    plt.close(fig)
    print(f"  [fig] {png.name} + {svg.name}")


def figure_roc_or_pr(kind, tag, tag_name):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, label in zip(axes, LABELS):
        for model in MODELS:
            y_true, y_prob = load_scores(model, label=label, tag=tag)
            if kind == "roc":
                x, y, _ = roc_curve(y_true, y_prob)
                ax.plot(x, y, color=COLORS[model], lw=1.2, label=f"{DISPLAY[model]} ({load_metrics(model, label=label, tag=tag)['roc_auc']:.3f})")
            else:
                p, r, _ = precision_recall_curve(y_true, y_prob)
                base = float(np.mean(y_true))
                ax.plot(r, p, color=COLORS[model], lw=1.2, label=f"{DISPLAY[model]} ({load_metrics(model, label=label, tag=tag)['pr_auc']:.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8, label="随机基准 (0.500)")
            ax.set_xlabel("假阳率 FPR")
            ax.set_ylabel("真阳率 TPR")
            ax.set_title(f"ROC 曲线 · {LABELS[label]}")
        else:
            ax.axhline(float(np.mean(y_true)), color="grey", ls="--", lw=0.8, label="正类占比基准")
            ax.set_xlabel("召回率 Recall")
            ax.set_ylabel("精确率 Precision")
            ax.set_title(f"PR 曲线 · {LABELS[label]}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left" if kind == "pr" else "lower right", fontsize=8.5, frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, f"fig_{tag_name}")


def figure_metrics_summary():
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.4))
    panels = [("labelA5d", "roc_auc", "A5d · ROC-AUC"), ("labelA5d", "pr_auc", "A5d · PR-AUC"),
              ("labelB4w", "roc_auc", "B4w · ROC-AUC"), ("labelB4w", "pr_auc", "B4w · PR-AUC")]
    x = np.arange(len(MODELS))
    for ax, (label, key, title) in zip(axes.ravel(), panels):
        vals = [load_metrics(m, label=label)[key] for m in MODELS]
        ax.bar(x, vals, color=[COLORS[m] for m in MODELS], width=0.62)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[m] for m in MODELS])
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.suptitle("gen5 面板（1,577 只）测试段 2025-01 ~ 2026-08：五模型普通配置对比", y=0.99)
    fig.tight_layout()
    save(fig, "fig_metrics_summary")


def figure_by_year():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, label in zip(axes, LABELS):
        years = sorted({int(y) for m in MODELS for y in load_metrics(m, label=label)["by_year"]})
        width = 0.8 / len(MODELS)
        for i, model in enumerate(MODELS):
            by_year = load_metrics(model, label=label)["by_year"]
            vals = [by_year.get(str(y), {}).get("roc_auc") or np.nan for y in years]
            ax.bar(np.arange(len(years)) + i * width - 0.4 + width / 2, vals, width=width, color=COLORS[model], label=DISPLAY[model])
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels([str(y) for y in years])
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.4, 0.95)
        ax.set_title(f"分年 ROC-AUC · {LABELS[label]}")
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.legend(fontsize=8.5, frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "fig_by_year_roc")


def figure_loss_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, label in zip(axes, LABELS):
        for model in ("lstm", "gru", "tcn"):
            with open(metrics_path(model, "gen5", label, ""), encoding="utf-8") as f:
                history = json.load(f)["loss_history"]
            ax.plot(range(1, len(history) + 1), history, color=COLORS[model], marker="o", ms=3, lw=1.1, label=DISPLAY[model])
        ax.set_xlabel("训练轮次 Epoch")
        ax.set_ylabel("训练损失 BCE")
        ax.set_title(f"训练损失曲线 · {LABELS[label]}")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_loss_curves")


def figure_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, label in zip(axes, LABELS):
        for model in MODELS:
            y_true, y_prob = load_scores(model, label=label, tag="_pw")
            bins = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(y_prob, bins[1:-1]), 0, 9)
            xs, ys = [], []
            for b in range(10):
                mask = idx == b
                if mask.any():
                    xs.append(y_prob[mask].mean())
                    ys.append(y_true[mask].mean())
            ax.plot(xs, ys, color=COLORS[model], marker="o", ms=3, lw=1.1, label=DISPLAY[model])
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8, label="理想校准")
        ax.set_xlabel("预测概率")
        ax.set_ylabel("实际正类频率")
        ax.set_title(f"概率校准 · {LABELS[label]}（加权配置）")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig_calibration_weighted")


def figure_threshold_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    thresholds = np.linspace(0.01, 0.99, 99)
    for ax, label in zip(axes, LABELS):
        for model in ("lstm", "tcn", "lgbm"):
            y_true, y_prob = load_scores(model, label=label)
            prec, rec = [], []
            for t in thresholds:
                pred = y_prob >= t
                tp = np.sum((pred == 1) & (y_true == 1))
                prec.append(tp / max(pred.sum(), 1))
                rec.append(tp / max(y_true.sum(), 1))
            ax.plot(thresholds, rec, color=COLORS[model], lw=1.2, ls="-", label=f"{DISPLAY[model]} Recall")
            ax.plot(thresholds, prec, color=COLORS[model], lw=1.1, ls=":", label=f"{DISPLAY[model]} Precision")
        ax.axvline(0.5, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("判定阈值 threshold")
        ax.set_ylabel("指标值")
        ax.set_ylim(0, 1)
        ax.set_title(f"阈值—精确/召回权衡 · {LABELS[label]}")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8.5, frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "fig_threshold_tradeoff")


def figure_scale_progression():
    gens = [("gen2", "40 只"), ("gen4", "662 只"), ("gen5", "1577 只")]
    sizes, roc_a, roc_b = [], [], []
    for gen, _ in gens:
        with open(MODELS_DIR / "lstm" / gen / "panel_stats.json", encoding="utf-8") as f:
            pool = json.load(f)["pool"]
        with open(MODELS_DIR / "lstm" / gen / "metrics_labelA5d_panel.json", encoding="utf-8") as f:
            roc_a.append(json.load(f)["metrics"]["roc_auc"])
        with open(MODELS_DIR / "lstm" / gen / "metrics_labelB4w_panel.json", encoding="utf-8") as f:
            roc_b.append(json.load(f)["metrics"]["roc_auc"])
        sizes.append(len(pool))

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(sizes, roc_a, marker="o", color=COLORS["lstm"], lw=1.3, label="A5d（普通）")
    ax.plot(sizes, roc_b, marker="s", color="#d62728", lw=1.3, label="B4w（普通）")
    for x, y in zip(sizes, roc_a):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5)
    for x, y in zip(sizes, roc_b):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([name for _, name in gens])
    ax.set_xlabel("股票池规模（历史成分并集）")
    ax.set_ylabel("测试 ROC-AUC")
    ax.set_ylim(0.5, 0.8)
    ax.set_title("LSTM 基线的面板规模效应")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    ax.text(0.02, 0.03, "注：gen2 为早期协议（train_ratio 切分、15 epochs），gen4/gen5 为统一协议",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_scale_progression")


def _box(ax, x, y, w, h, text, color="#eef3fb", edge="#3b5b87", fontsize=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0.9, edgecolor=edge, facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, p1, p2, color="#44546a"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=color,
                                 shrinkA=1, shrinkB=1))


def figure_pipeline():
    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    top = [
        (0.15, "BaoStock 日线\n(A 股后复权)", "#eaf3ea", "#3f7d3f"),
        (2.15, "股票池\n历史成分并集", "#eaf3ea", "#3f7d3f"),
        (4.15, "特征工程\n21 维 · 仅用 t 及以前", "#eaf3ea", "#3f7d3f"),
        (6.15, "标签工程\nA5d / B4w（Dimson）", "#fdf1e3", "#a06a1f"),
        (8.15, "周采样滑窗\n60 交易日 × 21 维", "#eaf3ea", "#3f7d3f"),
    ]
    for x, text, fc, ec in top:
        _box(ax, x, 3.75, 1.7, 0.95, text, fc, ec, fontsize=9)
    for x in (1.85, 3.85, 5.85, 7.85):
        _arrow(ax, (x, 4.22), (x + 0.3, 4.22))

    _box(ax, 0.15, 2.15, 2.3, 0.8, "按时间切分 + 6 周 purge\nscaler 仅在训练段拟合", "#f6ecf7", "#6a3b7d", fontsize=8.5)
    _box(ax, 2.95, 2.15, 3.5, 0.8, "模型（统一协议）\nLSTM / GRU / TCN / LightGBM / Logistic", "#e9eef5", "#33517a", fontsize=8.5)
    _box(ax, 6.95, 2.15, 2.9, 0.8, "测试评估：ROC-AUC / PR-AUC\nRecall / F1 + 分年 · 校准 · 阈值", "#e9eef5", "#33517a", fontsize=8.5)
    _arrow(ax, (2.45, 2.55), (2.95, 2.55))
    _arrow(ax, (6.45, 2.55), (6.95, 2.55))
    _arrow(ax, (9.0, 3.75), (9.0, 2.95))

    ax.text(0.15, 1.62, "面板 1,541 只有效标的 / 711,848 周样本（train 510,467 · test 124,590）", fontsize=8.5)
    ax.plot([0.15, 9.85], [1.15, 1.15], color="#44546a", lw=1.0)
    ax.plot([0.15, 7.05], [1.15, 1.15], color="#33517a", lw=5, alpha=0.35)
    ax.plot([7.65, 9.85], [1.15, 1.15], color="#a06a1f", lw=5, alpha=0.45)
    ax.plot([7.05, 7.65], [1.15, 1.15], color="white", lw=5)
    ax.text(3.4, 0.95, "训练期 2017-01-06 ~ 2024-11-15", ha="center", fontsize=8.5, color="#33517a")
    ax.text(8.75, 0.95, "测试期 2025-01-03 ~ 2026-08-28", ha="center", fontsize=8.5, color="#a06a1f")
    ax.text(7.35, 0.95, "purge", ha="center", fontsize=8, color="#666666")
    ax.set_title("数据—标签—特征—训练—评估 全流程（防泄漏设计）", pad=6)
    fig.tight_layout()
    save(fig, "fig_pipeline")


def figure_architectures():
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9))
    specs = [
        ("LSTM", ["输入 x_t (21)", "LSTM 层 ×2 (64)", "全连接 → logit", "sigmoid"], "#1f77b4"),
        ("GRU", ["输入 x_t (21)", "GRU 层 ×2 (64)", "全连接 → logit", "sigmoid"], "#ff7f0e"),
        ("TCN", ["输入 x_t (21)", "膨胀因果卷积 ×2 (64)", "全局末步池化", "全连接 → logit"], "#2ca02c"),
    ]
    for ax, (name, stages, color) in zip(axes, specs):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(name)
        n = len(stages)
        for i, stage in enumerate(stages):
            y = 0.86 - i * 0.22
            ax.add_patch(FancyBboxPatch((0.12, y - 0.085), 0.76, 0.17,
                                        boxstyle="round,pad=0.01,rounding_size=0.02",
                                        linewidth=0.9, edgecolor=color, facecolor=color + "1a"))
            ax.text(0.5, y, stage, ha="center", va="center", fontsize=8.5)
            if i < n - 1:
                ax.add_patch(FancyArrowPatch((0.5, y - 0.085), (0.5, y - 0.135), arrowstyle="-|>",
                                             mutation_scale=8, lw=0.8, color="#44546a"))
    fig.suptitle("三类深度学习基线结构对照（窗口 60 × 21，输出崩盘概率）", y=0.99)
    fig.tight_layout()
    save(fig, "fig_architectures")


def main():
    apply_style()
    print("生成图表 →", FIG_DIR)
    figure_roc_or_pr("roc", "", "roc_plain")
    figure_roc_or_pr("pr", "", "pr_plain")
    figure_roc_or_pr("pr", "_pw", "pr_weighted")
    figure_metrics_summary()
    figure_by_year()
    figure_loss_curves()
    figure_calibration()
    figure_threshold_tradeoff()
    figure_scale_progression()
    figure_pipeline()
    figure_architectures()
    print("完成。")


if __name__ == "__main__":
    main()
