"""讓 Claude 看 K 線圖判讀型態（圖像化確認，而非參數化）。

設計重點（對應先前討論的風險）：
- 結構化輸出：型態、信心分數、關鍵點位置、理由，方便程式驗證。
- 多次投票：同一張圖問 n 次取多數決，降低結果不穩定。
- 完整存檔：每次呼叫的圖片雜湊、提示詞版本、模型、回答都寫進 jsonl，可事後重現。
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import anthropic

MODEL = "claude-opus-5-5"
PROMPT_VERSION = "v1"

PROMPT = """這是一張匿名化的日 K 線圖（上方為 K 線，紅漲綠跌；下方為成交量）。
沒有價格刻度和日期，X 軸是 K 棒序號（0 起算）。

請只根據圖形判斷，圖中「最近、且已經完成」的是下列哪一種型態：
- double_bottom：W 底。下跌後出現兩個高度相近的低點，中間有一段反彈（頸線），最後價格向上突破頸線。
- head_and_shoulders_top：頭肩頂。上漲後出現三個高點，中間（頭）明顯最高，兩側（肩）高度相近，最後價格向下跌破頸線。
- none：沒有上述完成的型態（包含型態還沒突破／跌破的情況）。

回答要求：
- confidence：0 到 1 之間，代表你的把握程度。
- key_points：W 底給 [第一個低點, 頸線高點, 第二個低點] 的 K 棒序號；頭肩頂給 [左肩, 頭, 右肩] 的序號；none 給空陣列。
- reasoning：一到兩句話說明依據。"""

SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "enum": ["double_bottom", "head_and_shoulders_top", "none"]},
        "confidence": {"type": "number"},
        "key_points": {"type": "array", "items": {"type": "integer"}},
        "reasoning": {"type": "string"},
    },
    "required": ["pattern", "confidence", "key_points", "reasoning"],
    "additionalProperties": False,
}


def _ask_once(client: anthropic.Anthropic, image_b64: str) -> dict:
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        # 遇到安全分類器拒答時，由伺服器自動改用建議的備援模型
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"模型拒答：{response.stop_details}")
    text = next(b.text for b in response.content if b.type == "text")
    result = json.loads(text)
    result["served_by"] = response.model
    return result


def judge(image_path: str, n_votes: int = 3, log_path: str | None = None,
          client: anthropic.Anthropic | None = None) -> dict:
    """對一張圖問 n_votes 次，回傳多數決結果與一致率。"""
    client = client or anthropic.Anthropic()
    data = Path(image_path).read_bytes()
    image_b64 = base64.standard_b64encode(data).decode()
    sha = hashlib.sha256(data).hexdigest()

    votes = []
    for k in range(n_votes):
        try:
            v = _ask_once(client, image_b64)
        except anthropic.RateLimitError:
            time.sleep(20)
            v = _ask_once(client, image_b64)
        votes.append(v)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "image": image_path, "sha256": sha, "vote": k, "model": MODEL,
                    "prompt_version": PROMPT_VERSION, "response": v,
                }, ensure_ascii=False) + "\n")

    winner, count = Counter(v["pattern"] for v in votes).most_common(1)[0]
    best = max((v for v in votes if v["pattern"] == winner), key=lambda v: v["confidence"])
    return {
        "pattern": winner,
        # 投給其他型態的票視為信心 0，所以意見分歧時信心會自動打折
        "confidence": sum(v["confidence"] for v in votes if v["pattern"] == winner) / n_votes,
        "key_points": best["key_points"],
        "reasoning": best["reasoning"],
        "agreement": count / n_votes,
    }
