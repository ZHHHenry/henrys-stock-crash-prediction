import numpy as np
import pandas as pd


def daily_label_a(close: pd.Series, horizon: int = 5, threshold: float = -0.08) -> pd.Series:
    ret = close.astype(float).pct_change()
    future_min = ret.rolling(horizon).min().shift(-horizon)
    label = (future_min < threshold).astype(float)
    label[future_min.isna()] = np.nan
    return label


def build_weekly_returns(stock_close: pd.Series, index_close: pd.Series) -> pd.DataFrame:
    sw = stock_close.astype(float).resample("W-FRI").last().dropna()
    iw = index_close.astype(float).resample("W-FRI").last().dropna()
    both = pd.DataFrame({"stock": sw, "market": iw}).dropna()
    return both.pct_change().dropna()


def dimson_residuals(rets: pd.DataFrame, window: int = 104, min_obs: int | None = None) -> pd.Series:
    if min_obs is None:
        min_obs = max(int(window * 0.8), 40)
    market = rets["market"].astype(float)
    stock = rets["stock"].astype(float)
    design = pd.DataFrame(
        {
            "m-2": market.shift(2),
            "m-1": market.shift(1),
            "m0": market,
            "m+1": market.shift(-1),
            "m+2": market.shift(-2),
        }
    )
    design_values = design.values
    stock_values = stock.values
    n = len(stock)
    eps = np.full(n, np.nan)
    for t in range(n):
        lo = t - window + 1
        if lo < 0:
            continue
        xs = design_values[lo : t + 1]
        ys = stock_values[lo : t + 1]
        mask = ~np.isnan(xs).any(axis=1) & ~np.isnan(ys)
        if mask.sum() < min_obs:
            continue
        xt = design_values[t]
        if np.isnan(xt).any() or np.isnan(ys[-1]):
            continue
        matrix = np.column_stack([np.ones(int(mask.sum())), xs[mask]])
        beta, *_ = np.linalg.lstsq(matrix, ys[mask], rcond=None)
        eps[t] = ys[-1] - (beta[0] + xt @ beta[1:])
    return pd.Series(eps, index=stock.index, name="eps")


def crash_week_flags(
    eps: pd.Series,
    stat_window: int = 52,
    sigma: float = 3.2,
    clean_gap: int = 2,
    min_periods: int = 40,
) -> pd.Series:
    w = np.log(np.clip(1.0 + eps.astype(float).values, 1e-2, None))
    n = len(w)
    flags = np.full(n, np.nan)
    for t in range(n):
        if np.isnan(w[t]):
            continue
        lo = t - stat_window - clean_gap + 1
        hi = t - clean_gap + 1
        if lo < 0:
            continue
        window_vals = w[lo:hi]
        window_vals = window_vals[~np.isnan(window_vals)]
        if len(window_vals) < min_periods:
            continue
        mu = window_vals.mean()
        sd = window_vals.std(ddof=1)
        if sd <= 0:
            continue
        flags[t] = 1.0 if w[t] <= mu - sigma * sd else 0.0
    return pd.Series(flags, index=eps.index, name="crash_week")


def horizon_label(flags: pd.Series, weeks: int = 4) -> pd.Series:
    columns = [flags.shift(-k) for k in range(1, weeks + 1)]
    matrix = pd.concat(columns, axis=1)
    return matrix.max(axis=1, skipna=False)
