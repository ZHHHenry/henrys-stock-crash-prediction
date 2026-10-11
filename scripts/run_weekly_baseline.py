import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsmodels.data import fetch_ohlcv, load_ohlcv
from tsmodels.dataset import WeeklyCrashDataset
from tsmodels.features import FEATURE_COLS, compute_features
from tsmodels.labels import (
    build_weekly_returns,
    crash_week_flags,
    daily_label_a,
    dimson_residuals,
    horizon_label,
)
from tsmodels.models import LSTMBaseline
from tsmodels.train import compute_metrics, predict_proba, prepare_features, train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="周采样双标签崩盘预测基线（标准A：-8%/未来5日；标准B：市场模型崩盘周/未来4周）"
    )
    parser.add_argument("--ticker", default="sz.300750", help="BaoStock 代码如 sz.300750，或 yfinance 代码如 AAPL")
    parser.add_argument("--index", default="sh.000300", help="市场基准指数（BaoStock 代码）")
    parser.add_argument("--data-dir", default="data", help="数据目录（相对仓库根），如 data/dataset1")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--source", choices=["auto", "yfinance", "baostock"], default="auto")
    parser.add_argument("--refresh", action="store_true", help="强制重新下载数据")
    parser.add_argument("--seq-len", type=int, default=60, help="输入回看交易日数")
    parser.add_argument("--beta-window", type=int, default=104, help="Dimson 回归滚动窗口（周）")
    parser.add_argument("--stat-window", type=int, default=52, help="崩盘周统计窗口（周）")
    parser.add_argument("--sigma", type=float, default=3.2)
    parser.add_argument("--clean-gap", type=int, default=2, help="统计窗口与当周之间的干净期（周）")
    parser.add_argument("--horizon-weeks", type=int, default=4, help="标准B预测窗口（周）")
    parser.add_argument("--label-a-horizon", type=int, default=5, help="标准A预测窗口（交易日）")
    parser.add_argument("--label-a-threshold", type=float, default=-0.08)
    parser.add_argument("--model", default="lstm", help="模型类型（输出目录第一层）")
    parser.add_argument("--gen", default="gen1", help="代际（输出目录第二层）")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--embargo-weeks", type=int, default=6, help="切分边界清理（周）")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight", action="store_true", help="用负正样本比加权正类")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    out_dir = ROOT / "outputs" / "models" / args.model / args.gen
    out_dir.mkdir(parents=True, exist_ok=True)

    stock_path = fetch_ohlcv(args.ticker, args.start, args.end, data_dir, refresh=args.refresh, source=args.source)
    index_path = fetch_ohlcv(args.index, args.start, args.end, data_dir, refresh=args.refresh, source=args.source)
    stock = load_ohlcv(stock_path)
    index = load_ohlcv(index_path)

    feats = compute_features(stock, index_close=index["Close"]).replace([np.inf, -np.inf], np.nan)
    feats = feats.dropna(subset=FEATURE_COLS)
    print(f"[features] {args.ticker} 有效日行 {len(feats)}（{feats.index[0].date()} ~ {feats.index[-1].date()}），特征 {len(FEATURE_COLS)} 维")

    label_a_daily = daily_label_a(stock["Close"], horizon=args.label_a_horizon, threshold=args.label_a_threshold)
    weekly = build_weekly_returns(stock["Close"], index["Close"])
    eps = dimson_residuals(weekly, window=args.beta_window)
    flags = crash_week_flags(eps, stat_window=args.stat_window, sigma=args.sigma, clean_gap=args.clean_gap)
    label_b = horizon_label(flags, weeks=args.horizon_weeks)

    flags_valid = flags.dropna()
    yearly = flags_valid.astype(int).groupby(flags_valid.index.year).sum()
    print(f"[labelB] 崩盘周总数 {int(flags_valid.sum())}，分年: {yearly.to_dict()}")

    table = pd.DataFrame(
        {
            "labelA5d": label_a_daily.reindex(label_b.index, method="ffill"),
            "labelB4w": label_b,
        }
    ).dropna()
    positions = feats.index.searchsorted(table.index, side="right") - 1
    keep = positions >= (args.seq_len - 1)
    table = table[keep]
    positions = positions[keep]
    if len(table) < 40:
        raise RuntimeError(f"可用周样本仅 {len(table)} 个，不足以训练（可放宽 beta-window 或换更长历史标的）")
    print(
        f"[samples] 周样本 {len(table)}（{table.index[0].date()} ~ {table.index[-1].date()}）；"
        f"labelA5d 正率 {table['labelA5d'].mean():.2%}；labelB4w 正率 {table['labelB4w'].mean():.2%}"
    )

    n = len(table)
    k = int(n * args.train_ratio)
    train_idx = np.arange(0, max(k - args.embargo_weeks, 1))
    test_idx = np.arange(k, n)
    if len(test_idx) < 5 or len(train_idx) < 20:
        raise RuntimeError(f"切分后样本不足：train {len(train_idx)}，test {len(test_idx)}")
    last_train_pos = positions[train_idx[-1]]
    train_rows = feats.iloc[: last_train_pos + 1]
    feats_scaled, prep_params = prepare_features(train_rows, feats, FEATURE_COLS)
    with open(out_dir / f"prep_{args.ticker}.json", "w", encoding="utf-8") as f:
        json.dump(prep_params, f, ensure_ascii=False, indent=2)
    with open(out_dir / f"diagnostics_flags_{args.ticker}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "yearly_crash_weeks": {int(year): int(count) for year, count in yearly.items()},
                "n_crash_weeks": int(flags_valid.sum()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[split] train {len(train_idx)} 个（至 {table.index[train_idx[-1]].date()}，尾部 purge {args.embargo_weeks} 周）；"
        f"test {len(test_idx)} 个（{table.index[test_idx[0]].date()} 起）"
    )

    tag = "_pw" if args.pos_weight else ""
    for label_name in ("labelA5d", "labelB4w"):
        labels = table[label_name]
        ds = WeeklyCrashDataset(feats_scaled, FEATURE_COLS, table.index, labels, seq_len=args.seq_len)
        train_loader = DataLoader(Subset(ds, train_idx.tolist()), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(Subset(ds, test_idx.tolist()), batch_size=args.batch_size, shuffle=False)

        y_train = labels.values[train_idx]
        train_pos_rate = float(y_train.mean())
        pos_weight = None
        if args.pos_weight:
            if 0 < train_pos_rate < 1:
                ratio = float((1 - train_pos_rate) / train_pos_rate)
                pos_weight = torch.tensor([ratio], device=device)
                print(f"[{label_name}] pos_weight = {ratio:.2f}")
            else:
                print(f"[{label_name}] 训练段正率 {train_pos_rate:.2%}，无法计算 pos_weight，按未加权训练")

        model = LSTMBaseline(len(FEATURE_COLS), args.hidden_dim, args.num_layers).to(device)
        history = train_model(model, train_loader, args.epochs, args.lr, device, pos_weight)
        for epoch, loss in enumerate(history, 1):
            print(f"[{label_name}] epoch {epoch}/{args.epochs} loss={loss:.4f}")

        y_true, y_prob = predict_proba(model, test_loader, device)
        metrics = compute_metrics(y_true, y_prob, args.threshold)
        metrics["train_positive_rate"] = round(train_pos_rate, 4)
        print(f"[{label_name}] test 指标: {json.dumps(metrics, ensure_ascii=False)}")

        torch.save(model.state_dict(), out_dir / f"{args.model}_{label_name}_{args.ticker}{tag}.pt")
        report = {"args": vars(args), "loss_history": history, "metrics": metrics}
        with open(out_dir / f"metrics_{label_name}_{args.ticker}{tag}.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[save] 模型与指标已保存到 outputs/models/{args.model}/{args.gen}/")


if __name__ == "__main__":
    main()
