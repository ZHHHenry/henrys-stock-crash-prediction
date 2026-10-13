"""传统机器学习对照基线：逻辑回归（logreg）与 LightGBM（lgbm）。

复用面板缓存产物（features.pkl + prep_panel.json + panel_samples.csv），
不重新计算特征与标签；把 60×21 滑窗展平为 1260 维向量后训练，
以与深度基线相同的协议评估（时间切分 + purge、双标签、普通/加权对照、固定 seed）。

用法（仓库根目录）：

    python scripts\\run_classic_baseline.py --panel-dir outputs/models/lstm/gen5 --model lgbm --gen gen5 --split-date 2025-01-01 --train-start 2017-01-01
    python scripts\\run_classic_baseline.py --panel-dir outputs/models/lstm/gen5 --model lgbm --gen gen5 --split-date 2025-01-01 --train-start 2017-01-01 --pos-weight
    python scripts\\run_classic_baseline.py --panel-dir outputs/models/lstm/gen5 --model logreg --gen gen5 --split-date 2025-01-01 --train-start 2017-01-01
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsmodels import panel as panel_io
from tsmodels.features import FEATURE_COLS
from tsmodels.train import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="传统机器学习对照基线（逻辑回归 / LightGBM）")
    parser.add_argument("--panel-dir", default="outputs/models/lstm/gen5", help="源面板目录（含缓存与 prep）")
    parser.add_argument("--model", default="lgbm", choices=["logreg", "lgbm"])
    parser.add_argument("--gen", default="gen5")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--train-start", default="", help="训练样本起始日期（含），空为不限制")
    parser.add_argument("--split-date", default="", help="测试段起点，空则按 train-ratio 切分")
    parser.add_argument("--embargo-weeks", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight", action="store_true", help="按训练集负正比加权正类")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", default="labelA5d,labelB4w", help="要训练的标签，逗号分隔（用于断点续跑）")
    parser.add_argument("--max-iter", type=int, default=500, help="逻辑回归最大迭代")
    parser.add_argument("--n-estimators", type=int, default=300, help="LightGBM 树数量")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    return parser.parse_args()


def build_estimator(args, train_pos_rate: float):
    if args.model == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=args.max_iter,
            class_weight="balanced" if args.pos_weight else None,
            random_state=args.seed,
        )
    from lightgbm import LGBMClassifier

    ratio = float((1 - train_pos_rate) / train_pos_rate) if 0 < train_pos_rate < 1 else 1.0
    return LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_jobs=-1,
        random_state=args.seed,
        scale_pos_weight=ratio if args.pos_weight else 1.0,
        verbose=-1,
    )


def main():
    args = parse_args()
    panel_dir = Path(args.panel_dir)
    if not panel_dir.is_absolute():
        panel_dir = ROOT / panel_dir
    out_dir = ROOT / "outputs" / "models" / args.model / args.gen
    out_dir.mkdir(parents=True, exist_ok=True)

    feats_raw, panel, prep = panel_io.load_panel_cache(panel_dir, FEATURE_COLS)
    scaled = panel_io.scale_features(feats_raw, prep, FEATURE_COLS)
    windows = panel_io.build_windows(panel, scaled, FEATURE_COLS, args.seq_len)
    flat = windows.reshape(len(panel), -1)
    print(f"[data] 面板样本 {len(panel)} | 展平维度 {flat.shape[1]} | 来源 {panel_dir.relative_to(ROOT)}")

    split, train_mask, test_mask = panel_io.time_split(
        panel,
        split_date=args.split_date or None,
        train_ratio=args.train_ratio,
        embargo_weeks=args.embargo_weeks,
        train_start=args.train_start or None,
    )
    if int(test_mask.sum()) < 50 or int(train_mask.sum()) < 200:
        raise RuntimeError(f"切分后样本不足：train {int(train_mask.sum())}，test {int(test_mask.sum())}")
    print(
        f"[split] split_date={split.date()}；train {int(train_mask.sum())}（尾部 purge {args.embargo_weeks} 周）；"
        f"test {int(test_mask.sum())}"
    )

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    test_years = pd.DatetimeIndex(panel.loc[test_mask, "date"]).year.to_numpy()
    tag = "_pw" if args.pos_weight else ""

    for label_name in [s.strip() for s in args.labels.split(",") if s.strip()]:
        y = panel[label_name].to_numpy()
        train_pos_rate = float(y[train_idx].mean())
        print(f"[{label_name}] 训练正率 {train_pos_rate:.4%} | 开始训练 {args.model}")
        estimator = build_estimator(args, train_pos_rate)
        estimator.fit(flat[train_idx], y[train_idx])
        y_prob = estimator.predict_proba(flat[test_idx])[:, 1]
        y_true = y[test_idx]

        metrics = compute_metrics(y_true, y_prob, args.threshold)
        metrics["train_positive_rate"] = round(train_pos_rate, 4)
        by_year = {}
        for year in np.unique(test_years):
            mask = test_years == year
            if int(mask.sum()) >= 10:
                by_year[int(year)] = compute_metrics(y_true[mask], y_prob[mask], args.threshold)
        metrics["by_year"] = by_year
        print(f"[{label_name}] test 指标: {json.dumps(metrics, ensure_ascii=False)}")

        report = {
            "args": vars(args),
            "source_panel": str(panel_dir.relative_to(ROOT)),
            "estimator": args.model,
            "loss_history": None,
            "metrics": metrics,
        }
        with open(out_dir / f"metrics_{label_name}_panel{tag}.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        panel_io.save_scores(out_dir, f"scores_{label_name}_panel{tag}", y_true, y_prob, years=test_years)

    print(f"[save] 指标与预测分数已保存到 outputs/models/{args.model}/{args.gen}/")


if __name__ == "__main__":
    main()
