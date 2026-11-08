"""把 K 線畫成「匿名化、規格固定」的圖片，給 AI 看圖判讀。

匿名化：沒有股票代號、沒有日期、沒有價格數字，X 軸只標 K 棒序號（0 ~ N-1），
方便 AI 回報關鍵點位置，再由程式驗證。
規格固定：圖片大小、K 棒數量、配色都寫死在這裡——這些其實是「藏起來的參數」，
要改也要做敏感度測試。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG_SIZE = (10, 5.5)
DPI = 110
UP, DOWN = "#d62728", "#2ca02c"  # 台股習慣：紅漲綠跌


def render(df: pd.DataFrame, path: str) -> None:
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=FIG_SIZE, dpi=DPI, sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )
    x = range(len(df))
    for i, r in zip(x, df.itertuples()):
        color = UP if r.close >= r.open else DOWN
        ax.vlines(i, r.low, r.high, color=color, linewidth=0.8)
        ax.bar(i, abs(r.close - r.open) or 1e-6, bottom=min(r.open, r.close), width=0.7, color=color)
        axv.bar(i, r.volume, width=0.7, color=color, alpha=0.6)

    ax.set_yticklabels([])  # 隱藏價格刻度
    ax.tick_params(axis="y", length=0)
    axv.set_yticks([])
    axv.set_xticks(range(0, len(df), 10))
    axv.set_xlabel("bar index")
    ax.grid(alpha=0.2)
    ax.set_xlim(-1, len(df))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
