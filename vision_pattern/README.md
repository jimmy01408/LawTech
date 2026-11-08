# AI 看圖確認 K 線型態：雛型

比較兩種判讀 K 線型態的方式：

| | 規則判讀（參數化） | AI 看圖判讀（圖像化） |
|---|---|---|
| 做法 | ZigZag 找轉折點 + 數字門檻 | 把 K 線畫成圖片，讓 Claude 判讀 |
| 程式 | `rules.py` | `vision_judge.py` |

## 流程

```
synth.py         產生合成 K 線，埋入已知型態（W底／頭肩頂／無型態）＝標準答案
   ↓
render.py        畫成匿名化圖片：無代號、無日期、無價格刻度，X 軸只標 K 棒序號
   ↓
rules.py         規則判讀 ─┐
vision_judge.py  看圖判讀 ─┤→ run_experiment.py 和標準答案比較，輸出 report.md
```

用合成資料的原因是**有標準答案**，而且 AI **不可能看過這些走勢**，可以避開「模型記得歷史行情」造成的未來函數。

## 執行

```bash
pip install -r vision_pattern/requirements.txt

# 只跑規則判讀（不需要 API key）
python vision_pattern/run_experiment.py --n-per-class 10

# 用 Claude API 看圖，每張圖問 3 次取多數決（需要 ANTHROPIC_API_KEY）
python vision_pattern/run_experiment.py --judge claude --votes 3

# 匯入其他來源的看圖判讀（人工標註等）
python vision_pattern/run_experiment.py --judge file --judgments vision_pattern/examples/judgments_in_session.json
```

輸出在 `vision_pattern/output/`：圖片、`answer_key.csv`（標準答案）、`results.csv`、`report.md`，以及 API 模式下每次呼叫的完整紀錄 `vision_log.jsonl`。

## 報告內容

- **準確率與混淆矩陣**：兩種方法分別對標準答案。
- **規則的參數敏感度**：只改 ZigZag 門檻，看準確率怎麼變。
- **關鍵點驗證**：AI 說「第一個低點在第 36 根」，程式會回頭確認那裡真的是局部低點、高度關係也符合型態，用來抓「自信地亂講」。
- **信心校準**：AI 說信心 0.8 的判讀，是否真的比 0.6 的準。
- **多次投票一致率**：同一張圖問多次，答案是否穩定。
- **互補分析**：規則錯但 AI 對、AI 錯但規則對的數量。

## 第一次試跑結果（`examples/`）

30 張圖（每類 10 張），seed=42。看圖判讀是在 Claude Code 對話中，由 Claude 在**沒看答案**的情況下判讀前 15 張，不是透過 API 執行。

| | 規則判讀 | 看圖判讀 |
|---|---|---|
| 準確率（同一批 15 張） | 73.3% | 100% |
| 關鍵點驗證通過 | — | 11/11 |

- 規則判讀全部 30 張的準確率是 76.7%。ZigZag 門檻從 0.02 到 0.08，準確率在 56.7% 到 83.3% 之間跳動，**對參數很敏感**。
- 規則的錯誤主要有兩種：把隨機震盪誤判成 W 底（`examples/sample_rules_false_positive.png`），以及雜訊把轉折點切碎而漏抓頭肩頂（`examples/sample_head_and_shoulders.png`）。

### 這個結果「不能」代表什麼

- **樣本太少**：15 張圖，100% 的信賴區間仍然很寬。
- **合成型態太乾淨**：真實行情的型態更模糊、更常失敗，這組題目偏簡單。
- **判讀者不獨立**：產生合成資料的程式也是同一個 AI 寫的，它知道型態大約埋在哪裡。要做公平測試，應該讓 API 上的新對話判讀，或請人工標註。
- **只驗證「認得出型態」**：沒有驗證型態之後的價格是否真的照預期走，也就是**還沒驗證能不能賺錢**。

## 下一步

1. 用 `--judge claude` 跑完整 30 張以上，並開 `--votes 3` 看穩定度。
2. 增加難度：更大的雜訊、不完整的型態、「快成形但沒突破」的干擾題。
3. 畫圖規格敏感度：同一段資料換成 60／90／120 根 K 線、加或不加均線，看判讀是否改變。
4. 換成真實行情，只使用模型訓練資料截止日之後的資料，並且移除代號和日期。
5. 接上回測：型態確認之後進場，用規則化停損和部位大小，比較「只有粗篩」和「粗篩 + AI 確認」的績效差異。
