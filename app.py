import io
import zipfile
from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st

REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]

# -----------------------------
# Core logic (signal + backtest)
# -----------------------------
def compute_signal(df: pd.DataFrame, zf_min: float = 7.0, vol_multi: float = 2.0, idx: int = 3) -> pd.Series:
    """
    复刻通达信公式：
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

    XG = (YD & LY4 & HOLD3).fillna(False)
    return XG


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
    - 默认 t+1 开盘买入
    - 默认持有 hold_days 个交易日后（含买入日）收盘卖出
    - 简单止损止盈：若触发则按“触发当日收盘”卖出（保守口径可改）
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

    # 逐笔复利权益曲线最大回撤
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


# -----------------------------
# Data loading helpers
# -----------------------------
def normalize_df(df):
    """
    自动把常见A股字段名 → 统一成:
    date, open, high, low, close, volume
    """
    col_map = {}

    for c in df.columns:
        cl = c.lower()
        if cl in ["date", "trade_date", "datetime", "交易日期"]:
            col_map[c] = "date"
        elif cl in ["open", "open_price", "开盘", "开盘价"]:
            col_map[c] = "open"
        elif cl in ["high", "high_price", "最高", "最高价"]:
            col_map[c] = "high"
        elif cl in ["low", "low_price", "最低", "最低价"]:
            col_map[c] = "low"
        elif cl in ["close", "close_price", "收盘", "收盘价"]:
            col_map[c] = "close"
        elif cl in ["volume", "vol", "成交量"]:
            col_map[c] = "volume"

    df = df.rename(columns=col_map)

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df



def read_csv_bytes(b: bytes) -> pd.DataFrame:
    # 尝试 utf-8 / gbk
    for enc in ["utf-8", "utf-8-sig", "gbk"]:
        try:
            return pd.read_csv(io.BytesIO(b), encoding=enc)
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(b))


def load_universe_from_upload(upload) -> dict:
    """
    支持：
    - 单个 CSV：返回 {filename_without_ext: df}
    - ZIP：zip内多个csv -> {symbol: df}
    """
    name = upload.name.lower()
    raw = upload.getvalue()

    data_map = {}

    if name.endswith(".csv"):
        df = read_csv_bytes(raw)
        df = normalize_df(df)
        symbol = upload.name.rsplit(".", 1)[0]
        data_map[symbol] = df
        return data_map

    if name.endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(raw))
        for info in z.infolist():
            if info.filename.lower().endswith(".csv") and not info.is_dir():
                b = z.read(info.filename)
                df = read_csv_bytes(b)
                df = normalize_df(df)
                symbol = info.filename.split("/")[-1].rsplit(".", 1)[0]
                data_map[symbol] = df
        if not data_map:
            raise ValueError("ZIP 里没有找到任何 CSV 文件。")
        return data_map

    raise ValueError("只支持上传 CSV 或 ZIP(内含多个CSV)。")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="通达信形态回测器（合富中国式）", layout="wide")
st.title("通达信形态回测器：连阳 + 炸板异动阳线 + 3天缩量不破开盘价")

with st.expander("数据要求（点开看）", expanded=False):
    st.markdown(
        """
- 上传 **CSV**（单票）或 **ZIP**（多票，zip 内多个 csv）
- 每个 CSV 至少包含列：`date, open, high, low, close, volume`
- 日期按日线；建议用**前复权**数据（否则分红送转会扭曲形态）
        """
    )

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("① 上传数据")
    upload = st.file_uploader("选择 CSV 或 ZIP", type=["csv", "zip"])

with colB:
    st.subheader("② 参数（复制粘贴即可）")
    zf_min = st.number_input("ZF_MIN（异动阳线最低涨幅%）", value=7.0, min_value=0.0, step=0.5)
    vol_multi = st.number_input("VOL_MULTI（放量倍数×5日均量）", value=2.0, min_value=0.1, step=0.1)
    idx = st.number_input("IDX（异动阳线距离当前天数）", value=3, min_value=1, step=1)

    st.markdown("---")
    hold_days = st.number_input("持有天数（默认5）", value=5, min_value=1, step=1)
    entry = st.selectbox("进场方式", ["next_open", "next_close"], index=0)
    fee_bps = st.number_input("手续费（单边，bps）", value=10.0, min_value=0.0, step=1.0)
    slippage_bps = st.number_input("滑点（单边，bps）", value=5.0, min_value=0.0, step=1.0)

    st.markdown("---")
    use_sl = st.checkbox("启用止损", value=False)
    stop_loss = st.number_input("止损比例（如 0.06=6%）", value=0.06, min_value=0.0, step=0.01) if use_sl else None

    use_tp = st.checkbox("启用止盈", value=False)
    take_profit = st.number_input("止盈比例（如 0.2=20%）", value=0.20, min_value=0.0, step=0.01) if use_tp else None

run = st.button("🚀 开始回测", type="primary", use_container_width=True)

if run:
    if not upload:
        st.error("请先上传 CSV 或 ZIP。")
        st.stop()

    try:
        data_map = load_universe_from_upload(upload)
    except Exception as e:
        st.error(f"读取数据失败：{e}")
        st.stop()

    all_trades = []
    summary_rows = []

    st.info(f"已载入标的数：{len(data_map)}。开始计算信号与回测…")

    for symbol, df in data_map.items():
        sig = compute_signal(df, zf_min=zf_min, vol_multi=vol_multi, idx=int(idx))
        trades = backtest_single(
            df, sig,
            entry=entry,
            hold_days=int(hold_days),
            fee_bps=float(fee_bps),
            slippage_bps=float(slippage_bps),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        if not trades.empty:
            trades.insert(0, "symbol", symbol)
            all_trades.append(trades)

        stats = summarize_trades(trades)
        stats["symbol"] = symbol
        summary_rows.append(stats)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values(["n_trades", "win_rate"], ascending=False)

    st.success("回测完成 ✅")

    # Overall summary
    st.subheader("总体统计（全标的合并）")
    overall = summarize_trades(trades_df) if not trades_df.empty else {"n_trades": 0}
    st.json(overall)

    # Tables
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("分标的统计（summary）")
        st.dataframe(summary_df, use_container_width=True, height=420)
        st.download_button(
            "下载 summary.csv",
            data=summary_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="summary.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c2:
        st.subheader("逐笔交易明细（trades）")
        st.dataframe(trades_df, use_container_width=True, height=420)
        st.download_button(
            "下载 trades.csv",
            data=trades_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="trades.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Year stats
    st.subheader("按年份统计（entry year）")
    if trades_df.empty:
        st.warning("没有产生任何交易（n_trades=0）。可以尝试降低阈值或扩大数据范围。")
    else:
        trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year
        year_stats = trades_df.groupby("year", as_index=False).apply(
            lambda x: pd.Series(summarize_trades(x))
        ).reset_index(drop=True)
        st.dataframe(year_stats, use_container_width=True)
        st.download_button(
            "下载 year_stats.csv",
            data=year_stats.to_csv(index=False).encode("utf-8-sig"),
            file_name="year_stats.csv",
            mime="text/csv",
            use_container_width=True
        )

