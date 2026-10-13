import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsmodels.data import fetch_hs300_constituents, fetch_ohlcv, load_ohlcv
from tsmodels.features import FEATURE_COLS, compute_features
from tsmodels.labels import (
    build_weekly_returns,
    crash_week_flags,
    daily_label_a,
    dimson_residuals,
    horizon_label,
)
from tsmodels import panel as panel_io
from tsmodels.models import GRUBaseline, LSTMBaseline, TCNBaseline
from tsmodels.train import (
    compute_metrics,
    fit_pooled_scaler,
    predict_proba,
    train_model,
    transform_frame,
)

MODEL_REGISTRY = {"lstm": LSTMBaseline, "gru": GRUBaseline, "tcn": TCNBaseline}


def parse_args():
    parser = argparse.ArgumentParser(description="面板版双标签崩盘预测基线（股票池 + 汇总训练）")
    parser.add_argument("--tickers", default="", help="逗号分隔的标的列表，覆盖默认股票池")
    parser.add_argument("--pool-file", default="", help="股票清单文件（每行 code 或 code,name），替代 HS300 抽样")
    parser.add_argument("--data-dir", default="data", help="数据目录（相对仓库根），如 data/dataset2")
    parser.add_argument("--max-stocks", type=int, default=40)
    parser.add_argument("--index", default="sh.000300")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--refresh", action="store_true", help="强制重新下载数据")
    parser.add_argument("--only-fetch", action="store_true", help="只抓取数据与构建面板，不训练")
    parser.add_argument("--rebuild-cache", action="store_true", help="忽略中间缓存强制重建")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--beta-window", type=int, default=104)
    parser.add_argument("--stat-window", type=int, default=52)
    parser.add_argument("--sigma", type=float, default=3.2)
    parser.add_argument("--clean-gap", type=int, default=2)
    parser.add_argument("--horizon-weeks", type=int, default=4)
    parser.add_argument("--label-a-horizon", type=int, default=5)
    parser.add_argument("--label-a-threshold", type=float, default=-0.08)
    parser.add_argument("--model", default="lstm", choices=sorted(MODEL_REGISTRY), help="模型类型（lstm/gru/tcn）")
    parser.add_argument("--gen", default="gen2")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--labels", default="labelA5d,labelB4w", help="要训练/评估的标签，逗号分隔（用于断点续跑）")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--train-start", default="", help="训练样本起始日期（含），空为不限制")
    parser.add_argument("--split-date", default="", help="显式切分日期（测试段起点），空则按 train-ratio 切分")
    parser.add_argument("--embargo-weeks", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight", action="store_true", help="用负正样本比加权正类")
    parser.add_argument("--eval-only", action="store_true", help="仅载入已保存权重做推理并保存预测分数，不训练、不改写指标")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_pool(args):
    if args.tickers:
        return [t.strip() for t in args.tickers.split(",") if t.strip()], "tickers 参数"
    if args.pool_file:
        path = Path(args.pool_file)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise RuntimeError(f"股票清单文件不存在: {path}")
        codes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.lstrip("\ufeff").strip()
            if line and not line.startswith("#"):
                codes.append(line.split(",")[0].strip())
        return codes, f"清单文件 {args.pool_file}（{len(codes)} 行）"
    codes = sorted(fetch_hs300_constituents())
    if len(codes) > args.max_stocks:
        step = len(codes) / args.max_stocks
        codes = [codes[int(i * step)] for i in range(args.max_stocks)]
    return codes, "沪深300成分抽样"


def cache_meta(args, codes):
    return {
        "pool_hash": hashlib.md5(",".join(codes).encode("utf-8")).hexdigest(),
        "start": args.start,
        "end": args.end,
        "beta_window": args.beta_window,
        "stat_window": args.stat_window,
        "sigma": args.sigma,
        "clean_gap": args.clean_gap,
        "horizon_weeks": args.horizon_weeks,
        "label_a_horizon": args.label_a_horizon,
        "label_a_threshold": args.label_a_threshold,
        "seq_len": args.seq_len,
    }


def build_stock(code, stock, args, index):
    feats = compute_features(stock, index_close=index["Close"]).replace([np.inf, -np.inf], np.nan)
    feats = feats.dropna(subset=FEATURE_COLS)
    label_a_daily = daily_label_a(
        stock["Close"], horizon=args.label_a_horizon, threshold=args.label_a_threshold
    )
    weekly = build_weekly_returns(stock["Close"], index["Close"])
    eps = dimson_residuals(weekly, window=args.beta_window)
    flags = crash_week_flags(eps, stat_window=args.stat_window, sigma=args.sigma, clean_gap=args.clean_gap)
    label_b = horizon_label(flags, weeks=args.horizon_weeks)
    table = pd.DataFrame(
        {
            "labelA5d": label_a_daily.reindex(label_b.index, method="ffill"),
            "labelB4w": label_b,
        }
    ).dropna()
    positions = feats.index.searchsorted(table.index, side="right") - 1
    table = table[positions >= (args.seq_len - 1)]
    return feats, table, flags


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

    index_path = fetch_ohlcv(args.index, args.start, args.end, data_dir, refresh=args.refresh)
    index = load_ohlcv(index_path)

    pool, pool_desc = build_pool(args)
    print(f"[pool] 股票池 {len(pool)} 只（来源：{pool_desc}）：{', '.join(pool[:8])} ...")

    cache_dir = out_dir / "panel_cache"
    cache_features = cache_dir / "features.pkl"
    expected_meta = cache_meta(args, pool)
    use_cache = cache_features.exists() and (cache_dir / "meta.json").exists() and not args.rebuild_cache
    if use_cache:
        with open(cache_dir / "meta.json", encoding="utf-8") as f:
            use_cache = json.load(f) == expected_meta
        if not use_cache:
            print("[cache] 缓存与当前参数/股票池不一致，将重建")

    feats_raw = {}
    stock_stats = {}
    failures = []

    if use_cache:
        with open(cache_features, "rb") as f:
            feats_raw = pickle.load(f)
        panel = pd.read_csv(out_dir / "panel_samples.csv", parse_dates=["date"])
        with open(out_dir / "panel_stats.json", encoding="utf-8") as f:
            stats_obj = json.load(f)
        stock_stats = stats_obj.get("stocks", {})
        failures = stats_obj.get("failures", [])
        print(f"[cache] 载入缓存：{len(feats_raw)} 只特征、{len(panel)} 条面板样本、{len(failures)} 条失败/跳过")
    else:
        panel_records = []
        for i, code in enumerate(pool, 1):
            csv_path = data_dir / f"{code}.csv"
            before_mtime = csv_path.stat().st_mtime_ns if csv_path.exists() else -1
            try:
                path = fetch_ohlcv(code, args.start, args.end, data_dir, refresh=args.refresh)
                stock = load_ohlcv(path)
            except Exception as exc:
                failures.append({"ticker": code, "reason": f"下载失败: {exc}"})
                print(f"[{i}/{len(pool)}] {code} 下载失败: {exc}")
                time.sleep(1.0)
                continue
            downloaded = csv_path.exists() and csv_path.stat().st_mtime_ns != before_mtime
            try:
                feats, table, flags = build_stock(code, stock, args, index)
                if len(table) < 30:
                    failures.append({"ticker": code, "reason": f"有效周样本仅 {len(table)} 个（历史过短）"})
                    print(f"[{i}/{len(pool)}] {code} 跳过：仅 {len(table)} 个样本")
                    if downloaded:
                        time.sleep(1.0)
                    continue
                feats_raw[code] = feats
                for day, row in table.iterrows():
                    panel_records.append((day, code, float(row["labelA5d"]), float(row["labelB4w"])))
                flags_valid = flags.dropna()
                stock_stats[code] = {
                    "samples": int(len(table)),
                    "crash_weeks": int(flags_valid.sum()),
                    "labelA_rate": round(float(table["labelA5d"].mean()), 4),
                    "labelB_rate": round(float(table["labelB4w"].mean()), 4),
                    "first": str(table.index[0].date()),
                    "last": str(table.index[-1].date()),
                }
                print(
                    f"[{i}/{len(pool)}] {code} 样本 {len(table)} | 崩盘周 {int(flags_valid.sum())} | "
                    f"A正率 {table['labelA5d'].mean():.2%} | B正率 {table['labelB4w'].mean():.2%}"
                )
            except Exception as exc:
                failures.append({"ticker": code, "reason": f"特征/标签失败: {exc}"})
                print(f"[{i}/{len(pool)}] {code} 特征/标签失败: {exc}")
            if downloaded:
                time.sleep(1.0)

        panel = pd.DataFrame(panel_records, columns=["date", "ticker", "labelA5d", "labelB4w"])
        panel = panel.sort_values("date").reset_index(drop=True)
        panel.to_csv(out_dir / "panel_samples.csv", index=False)
        with open(out_dir / "panel_stats.json", "w", encoding="utf-8") as f:
            json.dump({"pool": pool, "stocks": stock_stats, "failures": failures}, f, ensure_ascii=False, indent=2)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_features, "wb") as f:
            pickle.dump({code: frame[FEATURE_COLS] for code, frame in feats_raw.items()}, f)
        with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(expected_meta, f, ensure_ascii=False, indent=2)
        print(f"[cache] 已写入 {cache_dir.relative_to(ROOT)}")

    total_crash_weeks = sum(s["crash_weeks"] for s in stock_stats.values())
    print(
        f"[panel] 成功 {len(stock_stats)} 只 / 失败或跳过 {len(failures)} 只；"
        f"总样本 {len(panel)}；崩盘周合计 {total_crash_weeks}"
    )
    print(f"[panel] labelA5d 正率 {panel['labelA5d'].mean():.2%} | labelB4w 正率 {panel['labelB4w'].mean():.2%}")

    if args.only_fetch:
        print(f"[save] 面板数据已保存到 outputs/models/{args.model}/{args.gen}/")
        return

    n = len(panel)
    split_date, train_mask, test_mask = panel_io.time_split(
        panel,
        split_date=args.split_date or None,
        train_ratio=args.train_ratio,
        embargo_weeks=args.embargo_weeks,
        train_start=args.train_start or None,
    )
    if int(test_mask.sum()) < 50 or int(train_mask.sum()) < 200:
        raise RuntimeError(f"切分后样本不足：train {int(train_mask.sum())}，test {int(test_mask.sum())}")

    train_end_date = panel.loc[train_mask, "date"].max()
    pooled_train = pd.concat(
        [frame.loc[frame.index <= train_end_date, FEATURE_COLS] for frame in feats_raw.values()],
        ignore_index=True,
    )
    quantile_low, quantile_high, scaler = fit_pooled_scaler(pooled_train, FEATURE_COLS)
    feats_scaled = {
        code: transform_frame(frame, FEATURE_COLS, quantile_low, quantile_high, scaler)
        for code, frame in feats_raw.items()
    }
    prep_params = {
        "features": list(FEATURE_COLS),
        "clip_lower": quantile_low.tolist(),
        "clip_upper": quantile_high.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(out_dir / "prep_panel.json", "w", encoding="utf-8") as f:
        json.dump(prep_params, f, ensure_ascii=False, indent=2)

    print(
        f"[split] split_date={split_date.date()}；train {int(train_mask.sum())}（尾部 purge {args.embargo_weeks} 周）；"
        f"test {int(test_mask.sum())}"
    )
    print(
        f"[split] train {panel.loc[train_mask, 'date'].min().date()} ~ {panel.loc[train_mask, 'date'].max().date()}；"
        f"test {panel.loc[test_mask, 'date'].min().date()} ~ {panel.loc[test_mask, 'date'].max().date()}"
    )

    xs = panel_io.build_windows(panel, feats_scaled, FEATURE_COLS, args.seq_len)
    X = torch.tensor(xs, dtype=torch.float32)
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    test_years = pd.DatetimeIndex(panel.loc[test_mask, "date"]).year.to_numpy()

    tag = "_pw" if args.pos_weight else ""
    for label_name in [s.strip() for s in args.labels.split(",") if s.strip()]:
        y = torch.tensor(panel[label_name].values.astype(np.float32))
        train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X[test_idx], y[test_idx]), batch_size=args.batch_size, shuffle=False)

        y_train = panel[label_name].values[train_idx]
        train_pos_rate = float(y_train.mean())
        model = MODEL_REGISTRY[args.model](len(FEATURE_COLS), args.hidden_dim, args.num_layers).to(device)
        weights_path = out_dir / f"{args.model}_{label_name}_panel{tag}.pt"

        if args.eval_only:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            history = None
            print(f"[{label_name}] eval-only：载入 {weights_path.name}，跳过训练")
        else:
            pos_weight = None
            if args.pos_weight and 0 < train_pos_rate < 1:
                ratio = float((1 - train_pos_rate) / train_pos_rate)
                pos_weight = torch.tensor([ratio], device=device)
                print(f"[{label_name}] pos_weight = {ratio:.2f}")
            history = train_model(model, train_loader, args.epochs, args.lr, device, pos_weight)
            print(f"[{label_name}] loss: " + ", ".join(f"{value:.4f}" for value in history))

        y_true, y_prob = predict_proba(model, test_loader, device)
        metrics = compute_metrics(y_true, y_prob, args.threshold)
        metrics["train_positive_rate"] = round(train_pos_rate, 4)
        by_year = {}
        for year in np.unique(test_years):
            mask = test_years == year
            if int(mask.sum()) >= 10:
                by_year[int(year)] = compute_metrics(y_true[mask], y_prob[mask], args.threshold)
        metrics["by_year"] = by_year
        print(f"[{label_name}] test 指标: {json.dumps(metrics, ensure_ascii=False)}")

        panel_io.save_scores(out_dir, f"scores_{label_name}_panel{tag}", y_true, y_prob, years=test_years)
        if args.eval_only:
            metrics_path = out_dir / f"metrics_{label_name}_panel{tag}.json"
            if metrics_path.exists():
                stored = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
                print(
                    f"[{label_name}] 与已存指标对照：roc {stored.get('roc_auc')} -> {metrics['roc_auc']} | "
                    f"pr {stored.get('pr_auc')} -> {metrics['pr_auc']}"
                )
        else:
            torch.save(model.state_dict(), weights_path)
            report = {"args": vars(args), "loss_history": history, "metrics": metrics}
            with open(out_dir / f"metrics_{label_name}_panel{tag}.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[save] 模型与指标已保存到 outputs/models/{args.model}/{args.gen}/")


if __name__ == "__main__":
    main()
