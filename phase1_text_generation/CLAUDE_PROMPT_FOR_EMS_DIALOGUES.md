# Claude Prompt: 生成 EMS Radio 對話（Phase 1 離線版）

在另一台機器用 Claude 執行此 prompt，完成後將輸出存成 `text_corpus.jsonl` 帶回本機，再執行 `corpus_collector.py` 整合。

---

## 步驟 1：複製以下整段給 Claude

```
你是一個 EMS（緊急醫療服務）無線電通訊專家。請生成 500 條真實的 EMS radio 對話文本，用於訓練語音辨識模型。

【輸出格式】
每行一條 JSON，格式如下（不要有額外說明或 markdown 標記）：
{"text": "對話內容", "source": "llm", "scenario": "情境類型", "chief_complaint": "主訴"}

【規則】
1. text：1-3 句話，模擬真實 radio 通訊，簡短、有時不完整
2. 使用 radio 術語：10-4, copy, roger, standby, en route, on scene
3. 使用 EMS 縮寫：ALS, BLS, CPR, BP, HR, ETA, PD, tac
4. 情境類型 (scenario)：dispatch_unit / patient_report / hospital_notification / mass_casualty
5. 主訴 (chief_complaint)：chest pain, cardiac arrest, stroke, MVC, overdose, fall, unconscious, breathing difficulty, maternity, trauma, illness 等
6. 可穿插 Virginia Beach 地區街道名、地標
7. 不要用 [x] 或佔位符，要完整可讀文本
8. 每條 text 至少 15 字，最多約 200 字

【範例】
{"text": "1623p respond for a cardiac 4845 cleveland street 71 year old male conscious breathing 14b copy", "source": "llm", "scenario": "dispatch_unit", "chief_complaint": "cardiac arrest"}
{"text": "rescue 15 en route ETA 5 minutes we have one BLS patient 35 year old male fall with possible hip fracture", "source": "llm", "scenario": "patient_report", "chief_complaint": "fall with injury"}

請直接輸出 500 條，每行一條 JSON，不要編號、不要前言後語。
```

---

## 步驟 2：取得 Claude 輸出

- 複製 Claude 回覆的純文字
- 存成 `text_corpus.jsonl`（每行一條 JSON）
- 若 Claude 加了 markdown 程式碼區塊，請刪除 ``` 等標記，只保留 JSON 行

---

## 步驟 3：帶回本機後執行

```bash
cd /media/meow/One\ Touch/ems_call/ems_audio_syn_pipeline

# 將 text_corpus.jsonl 放到 phase1_output/ 目錄

python3 phase1_text_generation/corpus_collector.py \
    --human_csv ../vb_ems_anotation/human_anotation_vb.csv \
    --llm_jsonl phase1_output/text_corpus.jsonl \
    --output phase1_output/combined_corpus.jsonl
```

---

## 若 Claude 一次無法生成 500 條

可分批請 Claude 生成，例如：
- 「請生成 100 條，格式同上」
- 重複 5 次，再把 5 次輸出合併成一個 `text_corpus.jsonl`
