import io
import re
import zipfile
import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Strategy: Signal + Backtest
# =========================
def compute_signal(df: pd.DataFrame, zf_min: float = 7.0, vol_multi: float = 2.0, idx: int = 3) -> pd.Series:
    """
    复刻通达信：
    { 主升浪预备形态：连阳 + 炸板异动阳线 + 3天缩量不破开盘价 }
    XG: YD AND LY4 AND HOLD3;

    df: 必须包含列 ['open','high','low','close','volume']，按日期升序
    返回：bool Series，True 表示“当前这根K线满足XG”
    """
    O = df["open"].astype(float)
    L = df["low"].astype(float)
    C = df["close"].astype(float)
    V = df["volume"].astype(float)

    # 1) 异动阳线（向前 idx 天）
    E_YANG = (C.shift(idx) > O.shift(idx))
    E_ZF = ((C.shift(idx) / O.shift(idx) - 1.0) * 100.0 >= zf_min)

    V_MA5 = V.rolling(5, min_periods=5).mean()
    E_VOL = (V.shift(idx) >= vol_multi * V_MA5.shift(idx))

    YD = E_YANG & E_ZF & E_VOL

    # 2) 异动阳线之前连续4天阳线
    LY4 = (
        (C.shift(idx + 1) > O.shift(idx + 1)) &
        (C.shift(idx + 2) > O.shift(idx + 2)) &
        (C.shift(idx + 3) > O.shift(idx + 3)) &
        (C.shift(idx + 4) > O.shift(idx + 4))
    )

    # 3) 异动阳线后3天：不破异动阳线开盘价 + 缩量
    O_YD = O.shift(idx)
    HOLD3 = (
        (L.shift(2) >= O_YD) & (V.shift(2) < V.shift(idx)) &
        (L.shift(1) >= O_YD) & (V.shift(1) < V.shift(idx)) &
        (L >= O_YD) & (V < V.shift(idx))
    )

    return (YD & LY4 & HOLD3).fillna(False)


def backtest_single(
    df: pd.DataFrame,
    signal: pd.Series,
    entry: str = "next_open",
    hold_days: int = 5,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> pd.DataFrame:
    """
    单标的逐笔回测：
    - 信号日 t
    - 默认 t+1 开盘买入 (next_open) 或 next_close
    - 默认持有 hold_days 个交易日后（含买入日）收盘卖出
    - 简单止损止盈：若触发则按“触发当日收盘”卖出
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    O = df["open"].astype(float)
    H = df["high"].astype(float)
    L = df["low"].astype(float)
    C = df["close"].astype(float)

    sig_idx = np.where(signal.values)[0]
    trades = []

    cost = (fee_bps + slippage_bps) / 10000.0  # 单边成本（手续费+滑点）

    for t in sig_idx:
        buy_i = t + 1
        if buy_i >= len(df):
            continue

        buy_px = O.iloc[buy_i] if entry == "next_open" else C.iloc[buy_i]

        sell_i = buy_i + hold_days - 1
        if sell_i >= len(df):
            continue

        win_slice = slice(buy_i, sell_i + 1)
        sell_px = C.iloc[sell_i]
        exit_reason = f"time_exit_{hold_days}d"

        # 止损
        if stop_loss is not None:
            stop_level = buy_px * (1.0 - stop_loss)
            lows = L.iloc[win_slice].values
            if np.nanmin(lows) <= stop_level:
                hit = int(np.where(lows <= stop_level)[0][0])
                sell_i = buy_i + hit
                sell_px = C.iloc[sell_i]
                exit_reason = f"stop_loss_{stop_loss:.2%}"

        # 止盈（若更早触发则覆盖）
        if take_profit is not None:
            take_level = buy_px * (1.0 + take_profit)
            highs = H.iloc[win_slice].values
            if np.nanmax(highs) >= take_level:
                hit = int(np.where(highs >= take_level)[0][0])
                cand_i = buy_i + hit
                if cand_i < sell_i:
                    sell_i = cand_i
                    sell_px = C.iloc[sell_i]
                    exit_reason = f"take_profit_{take_profit:.2%}"

        gross_ret = (sell_px / buy_px) - 1.0
        net_ret = gross_ret - 2 * cost  # 双边成本

        trades.append(
            {
                "signal_date": df["date"].iloc[t],
                "entry_date": df["date"].iloc[buy_i],
                "exit_date": df["date"].iloc[sell_i],
                "entry_px": float(buy_px),
                "exit_px": float(sell_px),
                "hold_days": int((sell_i - buy_i) + 1),
                "gross_ret": float(gross_ret),
                "net_ret": float(net_ret),
                "exit_reason": exit_reason,
            }
        )

    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"n_trades": 0}

    r = trades["net_ret"].astype(float).fillna(0.0)
    win_rate = float((r > 0).mean())

    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1
    mdd = float(dd.min()) if len(dd) else 0.0

    pos_sum = float(r[r > 0].sum())
    neg_sum = float((-r[r < 0]).sum())
    profit_factor = float(pos_sum / (neg_sum + 1e-12))

    return {
        "n_trades": int(len(trades)),
        "win_rate": win_rate,
        "avg_ret": float(r.mean()),
        "median_ret": float(r.median()),
        "p25_ret": float(r.quantile(0.25)),
        "p75_ret": float(r.quantile(0.75)),
        "max_ret": float(r.max()),
        "min_ret": float(r.min()),
        "profit_factor": profit_factor,
        "mdd_by_trades": mdd,
    }


# =========================
# Robust CSV reading + column normalization
# =========================
REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]


def _clean_columns(cols):
    cleaned = []
    for c in cols:
        if c is None:
            cleaned.append("")
            continue
        s = str(c)
        s = s.replace("\ufeff", "")       # remove BOM
        s = s.strip()
        s = re.sub(r"\s+", "", s)         # remove all spaces
        cleaned.append(s)
    return cleaned


def _try_read_csv_bytes(b: bytes) -> pd.DataFrame:
    # 尝试常见编码
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
        try:
            df = pd.read_csv(io.BytesIO(b), encoding=enc)
            return df
        except Exception as e:
            last_err = e
            continue
    # 再试一把自动推断
    try:
        return pd.read_csv(io.BytesIO(b))
    except Exception as e:
        raise last_err or e


def _guess_header_row(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    有些数据第一行不是表头（例如说明文字），这里尝试找真正的表头行。
    逻辑：如果当前列名几乎都是 'Unnamed' 或数字，尝试把第1行当表头重读。
    """
    cols = [str(c) for c in df_raw.columns]
    unnamed_ratio = sum(("unnamed" in c.lower()) for c in cols) / max(len(cols), 1)
    numeric_like = sum(re.fullmatch(r"\d+", c.strip()) is not None for c in cols) / max(len(cols), 1)

    if unnamed_ratio > 0.6 or numeric_like > 0.6:
        # 尝试把第一行作为header
        df2 = df_raw.copy()
        df2.columns = df2.iloc[0].astype(str).tolist()
        df2 = df2.iloc[1:].reset_index(drop=True)
        return df2

    return df_raw


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    自动把常见A股字段名 → 统一成:
    date, open, high, low, close, volume

    支持：
    - tushare: ts_code, trade_date, open, high, low, close, vol
    - akshare: 日期/开盘/收盘/最高/最低/成交量
    - 通达信/同花顺导出：交易日期/开盘价/最高价/最低价/收盘价/成交量...
    """
    df = _guess_header_row(df)

    # 清理列名（去BOM/空格/隐藏字符）
    df.columns = _clean_columns(df.columns)

    # 有些文件列名是中文，但带括号/单位，做一次“标准化key”
    def norm_key(s: str) -> str:
        s = str(s).lower()
        s = s.replace("\ufeff", "")
        s = s.strip()
        s = re.sub(r"\s+", "", s)
        # 去掉括号和常见单位
        s = re.sub(r"[()（）\[\]【】]", "", s)
        s = s.replace("price", "价")
        return s

    # 建立映射：原列名 -> 标准列名
    col_map = {}

    # 先构造一个 {normalized: original} 便于匹配
    norm_to_orig = {norm_key(c): c for c in df.columns}

    def map_first(target, candidates):
        for cand in candidates:
            cand_norm = norm_key(cand)
            # 直接命中 normalized
            if cand_norm in norm_to_orig:
                col_map[norm_to_orig[cand_norm]] = target
                return True
        return False

    # 日期
    map_first("date", [
        "date", "datetime", "time", "trade_date", "tradedate",
        "交易日期", "日期", "时间", "t", "日 期"
    ])

    # OHLC
    map_first("open",  ["open", "open_price", "开盘", "开盘价", "今开", "开盘价元"])
    map_first("high",  ["high", "high_price", "最高", "最高价", "最高价元"])
    map_first("low",   ["low", "low_price", "最低", "最低价", "最低价元"])
    map_first("close", ["close", "close_price", "收盘", "收盘价", "今收", "收盘价元"])

    # volume
    map_first("volume", [
        "volume", "vol", "v", "成交量", "成交量手", "成交量股", "成交量万股", "成交量万",
        "volume_shares", "volume股", "volume手"
    ])

    df = df.rename(columns=col_map)

    # 最后检查必须列是否齐全
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        # 给出更友好的提示：显示当前列名
        raise ValueError(
            f"缺少必要列: {missing}。当前识别到的列: {list(df.columns)[:50]}"
        )

    df = df[REQUIRED_COLS].copy()

    # 日期解析：支持 20250131 / 2025-01-31 / 2025/01/31
    # 如果是 int (yyyymmdd)，转成字符串再解析
    if np.issubdtype(df["date"].dtype, np.number):
        df["date"] = df["date"].astype("Int64").astype(str)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # 数值列转 float
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df


def _is_panel_data(df: pd.DataFrame) -> str | None:
    """
    判断是否是“一个CSV含多只股票”的面板数据：
    常见股票代码列：ts_code / symbol / code / 股票代码 / 证券代码
    返回找到的代码列名，否则 None
    """
    cols = list(df.columns)
    candidates = ["ts_code", "symbol", "code", "股票代码", "证券代码", "ticker"]
    for c in candidates:
        if c in cols:
            return c
    # 再做一次宽松匹配（去空格后）
    cols_clean = {re.sub(r"\s+", "", str(c)).lower(): c for c in cols}
    for c in candidates:
        k = re.sub(r"\s+", "", c).lower()
        if k in cols_clean:
            return cols_clean[k]
    return None


def load_universe_from_upload(upload) -> dict:
    """
    支持：
    - 单个 CSV：
        a) 单票数据 -> {filename: df}
        b) 面板数据（含多票代码列） -> {symbol: df_by_symbol}
    - ZIP：zip内多个csv -> {symbol: df}
    """
    name = upload.name.lower()
    raw = upload.getvalue()

    data_map = {}

    if name.endswith(".csv"):
        df = _try_read_csv_bytes(raw)

        # 清理列名（先做一次，以便识别面
