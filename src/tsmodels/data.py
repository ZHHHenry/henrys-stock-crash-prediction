import io
import socket
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

SOURCES = ("auto", "yfinance", "baostock")
STANDARD_COLS = ["Open", "High", "Low", "Close", "Volume", "Amount", "Turn"]
BAOSTOCK_FIELDS_FULL = "date,open,high,low,close,volume,amount,turn"
BAOSTOCK_FIELDS_BASIC = "date,open,high,low,close,volume"
BAOSTOCK_RENAME = {
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "amount": "Amount",
    "turn": "Turn",
}
BAOSTOCK_TIMEOUT = 30
BAOSTOCK_ATTEMPTS = 3
BAOSTOCK_RETRY_SLEEP = 5


def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str,
    out_dir: Path,
    refresh: bool = False,
    source: str = "auto",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.csv"
    if out_path.exists() and not refresh and _cache_ok(out_path):
        return out_path
    resolved = _resolve_source(ticker, source)
    if resolved == "baostock":
        df = _fetch_baostock(ticker, start, end)
    else:
        df = _fetch_yfinance(ticker, start, end)
    if df is None or df.empty:
        raise RuntimeError(f"{resolved} 未返回 {ticker} 的数据，请检查代码、日期范围或网络")
    df = df.reindex(columns=STANDARD_COLS)
    df.index.name = "Date"
    df.to_csv(out_path)
    return out_path


def _cache_ok(path: Path) -> bool:
    try:
        cols = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return False
    return all(col in cols for col in STANDARD_COLS)


def _resolve_source(ticker: str, source: str) -> str:
    if source != "auto":
        return source
    return "baostock" if "." in ticker else "yfinance"


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as exc:
        raise RuntimeError(f"yfinance 下载失败（可能被限流）: {exc}") from exc
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df["Amount"] = df["Close"] * df["Volume"]
    df["Turn"] = np.nan
    return df


def _baostock_login(bs):
    with redirect_stdout(io.StringIO()):
        return bs.login()


def _baostock_logout(bs):
    try:
        with redirect_stdout(io.StringIO()):
            bs.logout()
    except Exception:
        pass


def _baostock_query(bs, ticker: str, start: str, end: str, fields: str):
    with redirect_stdout(io.StringIO()):
        rs = bs.query_history_k_data_plus(
            ticker,
            fields,
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="1",
        )
        rows = []
        while rs.error_code == "0" and rs.next() and len(rows) < 500000:
            rows.append(rs.get_row_data())
    return rs.error_code, rs.error_msg, (rs.fields if rows else fields.split(",")), rows


def _fetch_baostock(ticker: str, start: str, end: str) -> pd.DataFrame:
    import baostock as bs

    socket.setdefaulttimeout(BAOSTOCK_TIMEOUT)
    last_error = ""
    for _ in range(BAOSTOCK_ATTEMPTS):
        try:
            login = _baostock_login(bs)
            if login.error_code != "0":
                last_error = f"登录失败: {login.error_msg}"
                time.sleep(BAOSTOCK_RETRY_SLEEP)
                continue
            try:
                code, msg, fields, rows = _baostock_query(bs, ticker, start, end, BAOSTOCK_FIELDS_FULL)
                if code != "0" or not rows:
                    code, msg, fields, rows = _baostock_query(bs, ticker, start, end, BAOSTOCK_FIELDS_BASIC)
            finally:
                _baostock_logout(bs)
            if code != "0" or not rows:
                last_error = f"查询失败: {msg}"
                time.sleep(BAOSTOCK_RETRY_SLEEP)
                continue
            df = pd.DataFrame(rows, columns=fields).rename(columns=BAOSTOCK_RENAME)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.dropna(subset=["Open", "High", "Low", "Close"])
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(BAOSTOCK_RETRY_SLEEP)
    raise RuntimeError(f"BaoStock 查询失败（已重试 {BAOSTOCK_ATTEMPTS} 次）: {last_error}")


def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col=0, parse_dates=True)


def fetch_hs300_constituents() -> list[str]:
    import baostock as bs

    socket.setdefaulttimeout(BAOSTOCK_TIMEOUT)
    last_error = ""
    for _ in range(BAOSTOCK_ATTEMPTS):
        try:
            login = _baostock_login(bs)
            if login.error_code != "0":
                last_error = f"登录失败: {login.error_msg}"
                time.sleep(BAOSTOCK_RETRY_SLEEP)
                continue
            try:
                with redirect_stdout(io.StringIO()):
                    rs = bs.query_hs300_stocks()
                    code_idx = rs.fields.index("code") if "code" in rs.fields else 1
                    codes = []
                    while rs.error_code == "0" and rs.next() and len(codes) < 500:
                        codes.append(rs.get_row_data()[code_idx])
            finally:
                _baostock_logout(bs)
            if rs.error_code == "0" and codes:
                return codes
            last_error = f"查询失败: {rs.error_msg}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(BAOSTOCK_RETRY_SLEEP)
    raise RuntimeError(f"BaoStock 沪深300成分查询失败（已重试 {BAOSTOCK_ATTEMPTS} 次）: {last_error}")
