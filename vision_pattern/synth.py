"""產生合成 K 線資料，並在其中埋入已知型態（作為標準答案）。

為什麼用合成資料：
- 有確定的標準答案（ground truth），才能衡量判讀準確率。
- AI 模型不可能在訓練資料裡「看過」這些走勢，避免記憶答案造成的未來函數。

型態類別：
- double_bottom          W 底（雙重底）
- head_and_shoulders_top 頭肩頂
- none                   無明確型態（隨機漫步）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PATTERNS = ["double_bottom", "head_and_shoulders_top", "none"]
N_BARS = 120


def _path_from_anchors(anchors: list[tuple[float, float]], n: int, rng, noise: float) -> np.ndarray:
    """anchors: (位置比例 0~1, 對數價格)。線性內插後加上平滑雜訊。"""
    xs = np.array([a[0] for a in anchors]) * (n - 1)
    ys = np.array([a[1] for a in anchors])
    base = np.interp(np.arange(n), xs, ys)
    eps = rng.normal(0, noise, n)
    smooth = np.zeros(n)
    for i in range(1, n):  # AR(1) 讓雜訊比較像真實價格的抖動
        smooth[i] = 0.6 * smooth[i - 1] + eps[i]
    return base + smooth


def _anchors(pattern: str, rng) -> list[tuple[float, float]]:
    j = lambda s: rng.uniform(-s, s)  # 小幅隨機擾動，讓每張圖都不同
    if pattern == "double_bottom":
        depth = rng.uniform(0.12, 0.22)       # 前段跌幅
        rebound = rng.uniform(0.06, 0.12)     # 兩底之間反彈幅度（頸線）
        l2_diff = j(0.02)                     # 第二個底和第一個底的差距
        return [
            (0.00, 0.0), (0.10 + j(0.03), -depth * 0.4),
            (0.30 + j(0.03), -depth),                       # 第一個底
            (0.48 + j(0.03), -depth + rebound),             # 頸線
            (0.68 + j(0.03), -depth + l2_diff),             # 第二個底
            (0.88 + j(0.02), -depth + rebound + 0.005),     # 回到頸線
            (1.00, -depth + rebound + rng.uniform(0.03, 0.07)),  # 突破
        ]
    if pattern == "head_and_shoulders_top":
        rise = rng.uniform(0.10, 0.18)
        head = rng.uniform(0.05, 0.09)
        neck = rise - rng.uniform(0.05, 0.08)
        rs_diff = j(0.02)
        return [
            (0.00, 0.0), (0.18 + j(0.03), rise),            # 左肩
            (0.32 + j(0.02), neck + j(0.01)),               # 左頸
            (0.48 + j(0.03), rise + head),                  # 頭
            (0.64 + j(0.02), neck + j(0.01)),               # 右頸
            (0.80 + j(0.03), rise + rs_diff),               # 右肩
            (0.92 + j(0.02), neck),
            (1.00, neck - rng.uniform(0.03, 0.06)),         # 跌破頸線
        ]
    # none：純隨機漫步，用細碎的錨點產生不規則走勢
    pts = np.cumsum(rng.normal(0, 0.035, 9))
    return [(i / 8, float(pts[i] - pts[0])) for i in range(9)]


def make_series(pattern: str, seed: int, n: int = N_BARS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0.006, 0.014)
    logp = _path_from_anchors(_anchors(pattern, rng), n, rng, noise)
    close = 100 * np.exp(logp)

    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0, noise * 0.5, n - 1))
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = body_hi * np.exp(np.abs(rng.normal(0, noise * 0.7, n)))
    low = body_lo * np.exp(-np.abs(rng.normal(0, noise * 0.7, n)))

    vol = rng.lognormal(mean=10, sigma=0.3, size=n)
    if pattern != "none":  # 突破／跌破時放量
        vol[-int(n * 0.1):] *= rng.uniform(1.5, 2.5)

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})
