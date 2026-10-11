import numpy as np
import pandas as pd

FEATURE_COLS = [
    "log_r1",
    "log_r5",
    "log_r20",
    "vol5",
    "vol20",
    "atr14n",
    "park10",
    "c_ma5",
    "c_ma20",
    "c_ma60",
    "pctb20",
    "hi252",
    "rsi14",
    "macdh",
    "vr5_20",
    "turn",
    "turn_ratio",
    "amtz20",
    "c_rv20",
    "beta60",
    "rs20",
]


def compute_features(df: pd.DataFrame, index_close: pd.Series | None = None) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = out["Volume"].astype(float)
    log_close = np.log(close)
    log_ret = log_close.diff()

    out["log_r1"] = log_ret
    out["log_r5"] = log_close.diff(5)
    out["log_r20"] = log_close.diff(20)
    out["vol5"] = log_ret.rolling(5).std()
    out["vol20"] = log_ret.rolling(20).std()

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out["atr14n"] = true_range.rolling(14).mean() / close
    out["park10"] = np.sqrt((np.log(high / low) ** 2).rolling(10).mean() / (4 * np.log(2)))

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    out["c_ma5"] = close / ma5
    out["c_ma20"] = close / ma20
    out["c_ma60"] = close / ma60
    std20 = close.rolling(20).std()
    out["pctb20"] = (close - (ma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)
    out["hi252"] = close / close.rolling(252, min_periods=120).max() - 1

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = (100 - 100 / (1 + rs)) / 100
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    out["macdh"] = (dif - dea) / close

    vma5 = volume.rolling(5).mean()
    vma20 = volume.rolling(20).mean()
    out["vr5_20"] = vma5 / vma20

    turn = _turn_series(out)
    out["turn"] = turn
    out["turn_ratio"] = turn.rolling(5).mean() / turn.rolling(20).mean()

    amount = _amount_series(out, close, volume)
    log_amount = np.log(amount.clip(lower=1))
    out["amtz20"] = (log_amount - log_amount.rolling(20).mean()) / log_amount.rolling(20).std()

    dvol = np.log(volume.clip(lower=1)).diff()
    out["c_rv20"] = log_ret.rolling(20).corr(dvol)

    if index_close is not None:
        idx = index_close.reindex(out.index).astype(float)
        idx_ret = np.log(idx).diff()
        cov = log_ret.rolling(60, min_periods=40).cov(idx_ret)
        var = idx_ret.rolling(60, min_periods=40).var()
        out["beta60"] = cov / var.replace(0, np.nan)
        out["rs20"] = log_close.diff(20) - idx_ret.rolling(20).sum()
    else:
        out["beta60"] = np.nan
        out["rs20"] = np.nan

    out[FEATURE_COLS] = out[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    return out


def _turn_series(out: pd.DataFrame) -> pd.Series:
    volume = out["Volume"].astype(float)
    proxy = volume / volume.rolling(252, min_periods=60).mean()
    if "Turn" in out.columns and out["Turn"].notna().any():
        return out["Turn"].astype(float).fillna(proxy)
    return proxy


def _amount_series(out: pd.DataFrame, close: pd.Series, volume: pd.Series) -> pd.Series:
    if "Amount" in out.columns and out["Amount"].notna().any():
        return out["Amount"].astype(float)
    return close * volume
