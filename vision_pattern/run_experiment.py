"""實驗流程：合成資料 → 匿名化圖片 → 規則判讀 vs AI 看圖判讀 → 和標準答案比較。

用法：
  # 1. 只產生資料集與規則判讀（不需要 API key）
  python vision_pattern/run_experiment.py --n-per-class 10

  # 2. 用 Claude API 看圖（需要 ANTHROPIC_API_KEY）
  python vision_pattern/run_experiment.py --judge claude --votes 3

  # 3. 匯入其他來源的看圖判讀（例如人工標註，或在對話中由 AI 判讀）
  python vision_pattern/run_experiment.py --judge file --judgments path/to/judgments.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import rules  # noqa: E402
from render import render  # noqa: E402
from synth import PATTERNS, make_series  # noqa: E402

LABEL_ZH = {"double_bottom": "W底", "head_and_shoulders_top": "頭肩頂", "none": "無型態"}


def build_dataset(n_per_class: int, seed: int, out: Path) -> pd.DataFrame:
    """產生圖片。檔名是打亂後的流水號，從檔名看不出答案。"""
    rng = np.random.default_rng(seed)
    items = [(p, int(rng.integers(1e9))) for p in PATTERNS for _ in range(n_per_class)]
    order = rng.permutation(len(items))
    (out / "charts").mkdir(parents=True, exist_ok=True)
    rows = []
    for new_id, idx in enumerate(order):
        pattern, s = items[idx]
        df = make_series(pattern, s)
        cid = f"chart_{new_id:03d}"
        render(df, str(out / "charts" / f"{cid}.png"))
        df.to_csv(out / "charts" / f"{cid}.csv", index=False)
        rows.append({"chart_id": cid, "truth": pattern, "seed": s})
    key = pd.DataFrame(rows).sort_values("chart_id")
    key.to_csv(out / "answer_key.csv", index=False)  # 標準答案和圖片分開存，判讀時不要看
    return key


def keypoints_valid(df: pd.DataFrame, pattern: str, kp: list[int], win: int = 4, tol: float = 0.05) -> bool | None:
    """檢查 AI 說的關鍵點是否真的存在：位置附近要是局部高／低點，且高度關係符合型態。"""
    if pattern == "none":
        return None
    if len(kp) != 3 or any(not 0 <= k < len(df) for k in kp) or kp != sorted(kp):
        return False

    def near_extreme(k: int, col: str, fn) -> float | None:
        lo, hi = max(0, k - win), min(len(df), k + win + 1)
        seg = df[col].iloc[lo:hi]
        local = df[col].iloc[max(0, k - 2 * win): min(len(df), k + 2 * win + 1)]
        return float(fn(seg)) if fn(seg) == fn(local) else None

    if pattern == "double_bottom":
        l1, nk, l2 = near_extreme(kp[0], "low", min), near_extreme(kp[1], "high", max), near_extreme(kp[2], "low", min)
        return None not in (l1, nk, l2) and abs(l1 - l2) / l1 <= tol and nk > max(l1, l2)
    ls, hd, rs = (near_extreme(k, "high", max) for k in kp)
    return None not in (ls, hd, rs) and hd > max(ls, rs) and abs(ls - rs) / ls <= tol


def confusion(truth: pd.Series, pred: pd.Series) -> pd.DataFrame:
    m = pd.crosstab(truth.map(LABEL_ZH), pred.map(LABEL_ZH), rownames=["實際"], colnames=["判讀"])
    names = [LABEL_ZH[p] for p in PATTERNS]
    return m.reindex(index=names, columns=names, fill_value=0)


def md_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "output"))
    ap.add_argument("--judge", choices=["none", "claude", "file"], default="none")
    ap.add_argument("--judgments", help="--judge file 時的 JSON：{chart_id: {pattern, confidence, key_points, reasoning}}")
    ap.add_argument("--votes", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)

    key = build_dataset(args.n_per_class, args.seed, out)
    charts = {cid: pd.read_csv(out / "charts" / f"{cid}.csv") for cid in key.chart_id}

    # --- 規則判讀 ---
    key["rules"] = [rules.detect(charts[c])["pattern"] for c in key.chart_id]

    # 規則的參數敏感度：只動 ZigZag 門檻
    sens = []
    for z in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
        p = dict(rules.PARAMS, zigzag_pct=z)
        pred = [rules.detect(charts[c], p)["pattern"] for c in key.chart_id]
        sens.append({"zigzag_pct": z, "準確率": round(float(np.mean(np.array(pred) == key.truth.to_numpy())), 3)})

    # --- 看圖判讀 ---
    vision = None
    if args.judge == "claude":
        from vision_judge import judge
        log = out / "vision_log.jsonl"
        vision = {c: judge(str(out / "charts" / f"{c}.png"), args.votes, str(log)) for c in key.chart_id}
    elif args.judge == "file":
        vision = json.loads(Path(args.judgments).read_text(encoding="utf-8"))

    report = [f"# 型態判讀實驗報告\n\n樣本數：{len(key)}（每類 {args.n_per_class}），seed={args.seed}\n"]
    acc_r = float((key.rules == key.truth).mean())
    report += ["## 規則判讀（參數化）", f"準確率：**{acc_r:.1%}**\n", md_table(confusion(key.truth, key.rules)),
               "\n### 參數敏感度（ZigZag 門檻）\n", md_table(pd.DataFrame(sens).set_index("zigzag_pct"))]

    if vision is not None:
        missing = [c for c in key.chart_id if c not in vision]
        if missing:
            print(f"警告：{len(missing)} 張圖沒有看圖判讀，跳過：{missing[:5]}")
        key = key[key.chart_id.isin(vision)].copy()
        key["vision"] = [vision[c]["pattern"] for c in key.chart_id]
        key["confidence"] = [float(vision[c].get("confidence", np.nan)) for c in key.chart_id]
        key["kp_valid"] = [keypoints_valid(charts[c], vision[c]["pattern"], vision[c].get("key_points", []))
                           for c in key.chart_id]
        acc_v = float((key.vision == key.truth).mean())
        acc_r2 = float((key.rules == key.truth).mean())
        agree = float((key.vision == key.rules).mean())

        report += ["\n## AI 看圖判讀（圖像化）", f"準確率：**{acc_v:.1%}**（同一批圖的規則準確率：{acc_r2:.1%}）",
                   f"和規則判讀的一致率：{agree:.1%}\n", md_table(confusion(key.truth, key.vision))]

        kp = key.kp_valid.dropna()
        if len(kp):
            report.append(f"\n關鍵點驗證：AI 回報的關鍵點中，有 {kp.mean():.1%} 經程式確認確實存在（{int(kp.sum())}/{len(kp)}）")
        if "agreement" in next(iter(vision.values())):
            ag = np.mean([vision[c]["agreement"] for c in key.chart_id])
            report.append(f"\n多次投票的平均一致率：{ag:.1%}")

        # 校準：信心高的判讀是否真的比較準
        if key.confidence.notna().any():
            bins = pd.cut(key.confidence, [0, 0.6, 0.8, 1.0001], labels=["< 0.6", "0.6–0.8", "≥ 0.8"], right=False)
            cal = key.assign(對=(key.vision == key.truth)).groupby(bins, observed=False)["對"].agg(["count", "mean"])
            cal.columns = ["樣本數", "實際準確率"]
            cal["實際準確率"] = cal["實際準確率"].map(lambda v: "-" if pd.isna(v) else f"{v:.0%}")
            cal.index.name = "信心區間"
            report += ["\n### 信心校準", md_table(cal)]

        # 兩種方法的互補：規則錯但 AI 對、AI 錯但規則對
        both = key.assign(r=key.rules == key.truth, v=key.vision == key.truth)
        report += ["\n### 兩種方法的互補",
                   f"- 兩者都對：{int((both.r & both.v).sum())}",
                   f"- 只有 AI 對：{int((~both.r & both.v).sum())}",
                   f"- 只有規則對：{int((both.r & ~both.v).sum())}",
                   f"- 兩者都錯：{int((~both.r & ~both.v).sum())}"]
        wrong = key[key.vision != key.truth]
        if len(wrong):
            report.append("\n### AI 判讀錯誤的圖")
            for r in wrong.itertuples():
                report.append(f"- `{r.chart_id}`：實際 {LABEL_ZH[r.truth]}，AI 判為 {LABEL_ZH[r.vision]}"
                              f"（信心 {r.confidence:.2f}）— {vision[r.chart_id].get('reasoning', '')}")

    key.to_csv(out / "results.csv", index=False)
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
