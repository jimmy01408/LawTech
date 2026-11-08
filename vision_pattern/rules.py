"""規則化（參數化）的型態偵測器，作為 AI 看圖的對照組。

做法：先用 ZigZag 找出轉折點，再用明確的數字條件判斷型態。
所有門檻都集中在 PARAMS，方便做參數敏感度測試。
"""
from __future__ import annotations

import pandas as pd

PARAMS = {
    "zigzag_pct": 0.04,       # 轉折至少要反向走 4% 才算新的轉折點
    "bottom_tol": 0.035,      # W 底兩個低點相差不超過 3.5%
    "min_rebound": 0.05,      # W 底中間反彈至少 5%
    "head_margin": 0.02,      # 頭要比兩肩高至少 2%
    "shoulder_tol": 0.05,     # 兩肩高度相差不超過 5%
}


def zigzag(df: pd.DataFrame, pct: float) -> list[tuple[int, float, str]]:
    """回傳轉折點 [(index, price, 'H'/'L'), ...]，用最高價找高點、最低價找低點。"""
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    pivots: list[tuple[int, float, str]] = []
    trend = None
    hi_i, hi, lo_i, lo = 0, highs[0], 0, lows[0]  # 尚未確定方向時，同時追蹤高低點
    ext_i, ext_p = 0, highs[0]
    for i, (h, l) in enumerate(zip(highs, lows)):
        if trend is None:
            if h > hi:
                hi_i, hi = i, h
            if l < lo:
                lo_i, lo = i, l
            if hi >= lo * (1 + pct) and hi_i > lo_i:
                pivots.append((lo_i, lo, "L"))
                trend, ext_i, ext_p = "up", hi_i, hi
            elif lo <= hi * (1 - pct) and lo_i > hi_i:
                pivots.append((hi_i, hi, "H"))
                trend, ext_i, ext_p = "down", lo_i, lo
        elif trend == "up":
            if h > ext_p:
                ext_i, ext_p = i, h
            elif l <= ext_p * (1 - pct):
                pivots.append((ext_i, ext_p, "H"))
                trend, ext_i, ext_p = "down", i, l
        else:
            if l < ext_p:
                ext_i, ext_p = i, l
            elif h >= ext_p * (1 + pct):
                pivots.append((ext_i, ext_p, "L"))
                trend, ext_i, ext_p = "up", i, h
    if trend is not None:  # 最後一段尚未確認的極值也列入
        pivots.append((ext_i, ext_p, "H" if trend == "up" else "L"))
    return pivots


def detect(df: pd.DataFrame, p: dict = PARAMS) -> dict:
    piv = zigzag(df, p["zigzag_pct"])
    last = df["close"].iloc[-1]

    # 由近往遠找，最近成立的型態優先
    for k in range(len(piv) - 3, -1, -1):
        a, b, c = piv[k : k + 3]
        if (a[2], b[2], c[2]) == ("L", "H", "L"):
            l1, neck, l2 = a[1], b[1], c[1]
            if (
                abs(l1 - l2) / l1 <= p["bottom_tol"]
                and neck >= max(l1, l2) * (1 + p["min_rebound"])
                and last > neck
            ):
                return {"pattern": "double_bottom", "key_points": [a[0], b[0], c[0]]}
    for k in range(len(piv) - 5, -1, -1):
        seq = piv[k : k + 5]
        if [s[2] for s in seq] == ["H", "L", "H", "L", "H"]:
            ls, t1, hd, t2, rs = (s[1] for s in seq)
            if (
                hd >= max(ls, rs) * (1 + p["head_margin"])
                and abs(ls - rs) / ls <= p["shoulder_tol"]
                and last < min(t1, t2)
            ):
                return {"pattern": "head_and_shoulders_top", "key_points": [seq[0][0], seq[2][0], seq[4][0]]}
    return {"pattern": "none", "key_points": []}
