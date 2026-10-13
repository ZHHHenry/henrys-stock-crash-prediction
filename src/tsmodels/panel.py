"""面板数据的复用工具：缓存载入、标准化、滑窗构建、时间切分与预测分数保存。

面板构建产物（由 scripts/run_panel_baseline.py 生成）位于
outputs/models/<模型>/<代际>/：

- panel_cache/features.pkl：逐标的原始特征（列 FEATURE_COLS），与 index 对齐
- panel_cache/meta.json：构建参数指纹（用于缓存一致性校验）
- panel_samples.csv：面板样本表（date/ticker/labelA5d/labelB4w）
- panel_stats.json：逐标的统计与失败清单
- prep_panel.json：winsorize 边界与标准化参数（仅在训练段拟合）

对照实验（传统基线等）通过本模块复用上述产物，避免重复计算特征与标签。
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_panel_cache(out_dir, feature_cols):
    """载入面板缓存：返回 (逐标的特征表, 面板样本表, 预处理参数)。"""
    out_dir = Path(out_dir)
    with open(out_dir / "panel_cache" / "features.pkl", "rb") as f:
        feats_raw = pickle.load(f)
    panel = pd.read_csv(out_dir / "panel_samples.csv", parse_dates=["date"])
    with open(out_dir / "prep_panel.json", encoding="utf-8") as f:
        prep = json.load(f)
    missing = [col for col in feature_cols if col not in prep["features"]]
    if missing:
        raise ValueError(f"prep_panel.json 缺少特征列: {missing}")
    return feats_raw, panel, prep


def scale_features(feats_raw, prep, feature_cols):
    """按 prep_panel.json 的 winsorize 边界与标准化参数变换特征（列顺序对齐）。"""
    order = [prep["features"].index(col) for col in feature_cols]
    low = np.asarray(prep["clip_lower"], dtype=np.float64)[order]
    high = np.asarray(prep["clip_upper"], dtype=np.float64)[order]
    mean = np.asarray(prep["mean"], dtype=np.float64)[order]
    scale = np.asarray(prep["scale"], dtype=np.float64)[order]
    scaled = {}
    for code, frame in feats_raw.items():
        values = frame[feature_cols].to_numpy(dtype=np.float64)
        transformed = (np.clip(values, low, high) - mean) / scale
        scaled[code] = pd.DataFrame(transformed, index=frame.index, columns=feature_cols)
    return scaled


def build_windows(panel, scaled, feature_cols, seq_len):
    """把面板样本展开为固定长度滑窗张量，返回 (n, seq_len, len(feature_cols))。"""
    n_features = len(feature_cols)
    windows = np.zeros((len(panel), seq_len, n_features), dtype=np.float32)
    matrices = {code: frame[feature_cols].to_numpy(dtype=np.float32) for code, frame in scaled.items()}
    indices = {code: frame.index for code, frame in scaled.items()}
    for i, (date, ticker) in enumerate(zip(panel["date"], panel["ticker"])):
        pos = indices[ticker].searchsorted(date, side="right") - 1
        windows[i] = matrices[ticker][pos - seq_len + 1 : pos + 1]
    return windows


def time_split(panel, split_date=None, train_ratio: float = 0.8, embargo_weeks: int = 6, train_start=None):
    """按时间切分面板：返回 (split_date, train_mask, test_mask)。"""
    if split_date:
        split = pd.Timestamp(split_date)
    else:
        k = int(len(panel) * train_ratio)
        split = panel.loc[k, "date"]
    purge = pd.Timedelta(weeks=embargo_weeks)
    train_mask = (panel["date"] < split - purge).to_numpy()
    test_mask = (panel["date"] >= split).to_numpy()
    if train_start:
        train_mask = train_mask & (panel["date"] >= pd.Timestamp(train_start)).to_numpy()
    return split, train_mask, test_mask


def save_scores(out_dir, stem, y_true, y_prob, years=None):
    """保存测试集预测分数（供 ROC/PR/校准等曲线绘制），npz 压缩存储。"""
    payload = {
        "y_true": np.asarray(y_true, dtype=np.int8),
        "y_prob": np.asarray(y_prob, dtype=np.float32),
    }
    if years is not None:
        payload["years"] = np.asarray(years, dtype=np.int32)
    path = Path(out_dir) / f"{stem}.npz"
    np.savez_compressed(path, **payload)
    return path
