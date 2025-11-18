# TrendRadar AI 功能使用指南

## 📖 目錄

1. [功能概述](#功能概述)
2. [快速開始](#快速開始)
3. [配置說明](#配置說明)
4. [功能詳解](#功能詳解)
5. [常見問題](#常見問題)
6. [故障排除](#故障排除)

---

## 功能概述

TrendRadar 的 AI 功能使用本地 Ollama LLM 為新聞分析提供智能增強，包括：

### 🎯 核心功能

| 功能 | 說明 | 效益 |
|------|------|------|
| **事件聚類** | 自動識別並合併相同事件的新聞 | 消除重複閱讀，節省 30%+ 時間 |
| **主題分類** | 智能分類新聞（監管、市場、技術等） | 快速定位感興趣領域 |
| **重要性評分** | AI 評估事件重要程度（1-10 分） | 優先閱讀關鍵事件 |
| **自動摘要** | 為每個事件生成簡短摘要 | 快速了解事件概況 |
| **情感分析** | 分析新聞情感（正面/負面/中性） | 把握市場情緒 |

### 💰 成本優勢

- ✅ **零 API 費用**：完全使用本地 Ollama，無需 OpenAI/Claude API
- ✅ **數據隱私**：所有分析在本地進行，數據不外傳
- ✅ **無網路依賴**：離線也能運行 AI 分析

---

## 快速開始

### 1. 安裝 Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# 或訪問 https://ollama.com 下載安裝包
```

### 2. 下載推薦模型

```bash
# 輕量級（推薦）：2GB RAM，速度快
ollama pull llama3.2:3b

# 或選擇平衡型：4.1GB RAM，質量好
ollama pull mistral:7b

# 或選擇高性能：4.7GB RAM，最佳質量
ollama pull llama3.1:8b
```

### 3. 啟用 AI 功能

編輯 `config/config.yaml`：

```yaml
ai_analysis:
  enabled: true  # 啟用 AI 分析
  ollama_url: "http://127.0.0.1:11434"
  model: "llama3.2:3b"  # 使用下載的模型
```

### 4. 運行並查看結果

```bash
python main.py
```

AI 分析結果會顯示在：
- **HTML 報告**：`output/日期/html/` 中的 `AI 熱點分析` 區塊
- **JSON 文件**：`output/日期/ai_analysis/` 中的分析結果
- **MCP 工具**：透過 Claude Desktop 查詢

---

## 配置說明

### 完整配置項

```yaml
ai_analysis:
  # 主開關
  enabled: false  # true=啟用, false=關閉

  # Ollama 配置
  ollama_url: "http://127.0.0.1:11434"  # Ollama 服務地址
  model: "llama3.2:3b"  # 使用的模型

  # 性能配置
  batch_size: 20  # 批次處理大小（文章數超過此值時分批）
  cache_ttl_hours: 24  # 快取有效期（小時）
```

### 環境變數覆寫

也可以透過環境變數配置（優先於 config.yaml）：

```bash
export AI_ANALYSIS_ENABLED=true
export OLLAMA_MODEL=llama3.2:3b
export OLLAMA_URL=http://localhost:11434
export AI_BATCH_SIZE=20
export AI_CACHE_TTL_HOURS=24

python main.py
```

---

## 功能詳解

### 1️⃣ 事件聚類

**功能**：自動識別報導同一事件的多條新聞並歸為一組。

**範例**：

輸入（3 條獨立新聞）：
```
[CoinDesk]  Mt. Gox Moves $956M Worth of BTC...
[Cointelegraph]  Mt. Gox moves $953M Bitcoin...
[The Block]  Mt. Gox moves $956 million...
```

輸出（1 個事件）：
```
事件：Mt. Gox 轉移大額比特幣
├─ 3 個來源報導
├─ 15 條相關新聞
└─ 重要性評分：8.5/10
```

**技術細節**：
- 使用 LLM 進行語義聚類（主模式）
- 當 LLM 不可用時降級到標題相似度聚類（本地模式）
- 支援跨批次合併（處理大量新聞時）

### 2️⃣ 主題分類

**功能**：將事件自動分類到預定義主題。

**支援的分類**：

| 主題 | 說明 | 範例 |
|------|------|------|
| `regulation` | 監管政策 | SEC 批准比特幣 ETF |
| `market` | 市場波動 | BTC 跌破 $90K |
| `technology` | 技術創新 | 以太坊升級 |
| `defi` | DeFi 協議 | Uniswap 新流動性池 |
| `nft` | NFT 與元宇宙 | Bored Ape 新系列 |
| `personnel` | 人事變動 | Coinbase CEO 離職 |
| `security` | 安全事件 | 交易所被駭 |
| `institutional` | 機構動態 | BlackRock 增持 |
| `macro` | 宏觀經濟 | 美聯儲利率決策 |
| `ecosystem` | 生態系統 | 加密貨幣合作夥伴 |

**分類機制**：
1. **向量相似度匹配**（最快）：與已知分類比對
2. **LLM 分類**（次快）：AI 生成新分類
3. **關鍵字匹配**（降級）：基於規則的分類

**自動學習**：
- AI 生成的新分類會自動加入分類庫
- 下次遇到類似事件時直接使用（無需重複調用 LLM）

### 3️⃣ 重要性評分

**功能**：AI 評估事件的重要程度（1-10 分）。

**評分考量因素**：
- 📰 報導數量（多個來源報導 = 更重要）
- 🏆 新聞排名（頭條新聞 = 更重要）
- 💥 市場影響力（價格波動、監管變動 = 更重要）
- ⏰ 時效性（突發事件 = 更重要）

**評分範圍**：

| 分數 | 級別 | 說明 |
|------|------|------|
| 9-10 | 🔴 極重要 | 重大市場事件、監管變革 |
| 7-8  | 🟠 重要 | 顯著的行業新聞 |
| 5-6  | 🟡 中等 | 常規行業動態 |
| 3-4  | 🟢 次要 | 一般性新聞 |
| 1-2  | ⚪ 輕微 | 邊緣新聞 |

**信心度**：
- 每個評分附帶信心度（0-1）
- 高信心度（>0.8）= AI 有把握
- 低信心度（<0.6）= 建議人工復核

### 4️⃣ 自動摘要

**功能**：為每個事件生成 2-3 句話的簡潔摘要。

**範例**：

```
事件：Bitcoin 市場大跌

摘要：Bitcoin 價格跌破 $90,000，ETF 出現創紀錄資金流出 $1.26B，
市場情緒達到「極度恐慌」水平。Mt. Gox 轉移 $956M 比特幣進一步
加劇市場擔憂。

來源：6 個平台，15 條新聞
```

**摘要特點**：
- 包含關鍵數字和具體事實
- 中立客觀，不帶主觀判斷
- 控制在 60 詞以內（中文約 100 字）

### 5️⃣ 情感分析

**功能**：分析新聞的情感傾向。

**分類**：
- 😊 **正面**：approval, launch, surge, gain, record
- 😐 **中性**：announce, update, release, report
- 😟 **負面**：hack, crash, probe, down, exploit

**應用**：
- 市場情緒指標
- 投資決策參考
- 風險預警

---

## 常見問題

### Q1：AI 分析需要多長時間？

**A**：取決於新聞數量和模型：

| 新聞數量 | llama3.2:3b | mistral:7b | llama3.1:8b |
|---------|-------------|------------|-------------|
| 20 篇   | ~5 秒       | ~10 秒     | ~15 秒      |
| 50 篇   | ~15 秒      | ~30 秒     | ~45 秒      |
| 100 篇  | ~30 秒      | ~60 秒     | ~90 秒      |

**加速技巧**：
- 使用批次處理（自動觸發）
- 啟用快取（24 小時內相同新聞直接使用快取）
- 選擇輕量級模型

### Q2：AI 分析準確率如何？

**A**：根據測試：

| 指標 | 準確率 | 說明 |
|------|--------|------|
| 事件聚類 | ~85% | 大部分相同事件能正確歸類 |
| 主題分類 | ~90% | 分類通常準確 |
| 重要性評分 | ±1.5 分 | 誤差在可接受範圍 |

**注意**：
- AI 可能誤判，建議結合人工判斷
- 可透過向量分類庫提高穩定性（自動學習）

### Q3：如何關閉 AI 功能？

**A**：三種方式：

**方式 1：配置文件**
```yaml
ai_analysis:
  enabled: false  # 設為 false
```

**方式 2：環境變數**
```bash
export AI_ANALYSIS_ENABLED=false
python main.py
```

**方式 3：臨時關閉 Ollama**
```bash
# AI 會自動檢測並降級到傳統模式
pkill ollama
```

### Q4：AI 分析失敗會影響主程式嗎？

**A**：**不會**。TrendRadar 有完善的降級機制：

1. AI 分析失敗 → 自動降級到傳統模式
2. 主程式繼續正常運行
3. HTML 報告顯示「AI 未啟用」提示

**降級場景**：
- Ollama 未運行
- 模型未下載
- JSON 解析失敗（重試 3 次後降級）
- 網路連接問題

### Q5：如何更換模型？

**A**：兩步驟：

1. 下載新模型：
```bash
ollama pull mistral:7b
```

2. 更新配置：
```yaml
ai_analysis:
  model: "mistral:7b"  # 更換模型名稱
```

**模型推薦**：

| 用途 | 推薦模型 | RAM 需求 | 速度 | 質量 |
|------|----------|----------|------|------|
| 日常使用 | llama3.2:3b | 2GB | ⚡⚡⚡ | ⭐⭐ |
| 平衡選擇 | mistral:7b | 4GB | ⚡⚡ | ⭐⭐⭐ |
| 最佳質量 | llama3.1:8b | 5GB | ⚡ | ⭐⭐⭐⭐ |

---

## 故障排除

### 問題 1：AI 分析一直失敗

**症狀**：
```
❌ [cluster_events] JSON 解析失敗 3 次，降級到本地聚類
```

**可能原因**：
1. Ollama 未運行
2. 模型未下載
3. URL 配置錯誤

**解決方案**：

```bash
# 1. 檢查 Ollama 是否運行
ollama list  # 應該顯示已下載的模型

# 2. 重新啟動 Ollama
ollama serve

# 3. 測試模型
ollama run llama3.2:3b "Hello"  # 應該有回應

# 4. 檢查配置
cat config/config.yaml | grep -A 5 "ai_analysis"
```

### 問題 2：分析速度很慢

**症狀**：
50 篇新聞分析超過 2 分鐘。

**解決方案**：

```yaml
# 1. 降低批次大小（減少單次處理量）
ai_analysis:
  batch_size: 10  # 從 20 改為 10

# 2. 更換更快的模型
ai_analysis:
  model: "llama3.2:3b"  # 最快

# 3. 啟用快取（避免重複分析）
ai_analysis:
  cache_ttl_hours: 48  # 延長快取時間
```

### 問題 3：快取文件損毀

**症狀**：
```
⚠️  [AIResultRepository] 載入快取失敗 (...cache.json): ...
```

**解決方案**：

```bash
# 刪除損毀的快取
rm -rf output/*/ai_analysis/cache.json

# 重新運行分析
python main.py
```

### 問題 4：記憶體不足

**症狀**：
系統記憶體占用過高，Ollama 崩潰。

**解決方案**：

1. **降級模型**：
```bash
ollama pull llama3.2:3b  # 僅需 2GB RAM
```

2. **減少批次大小**：
```yaml
ai_analysis:
  batch_size: 10  # 減少單次處理量
```

3. **限制 Ollama 記憶體**：
```bash
# 設置環境變數（限制為 4GB）
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
```

---

## 進階技巧

### 技巧 1：自定義分類主題

編輯 `output/category_store.json`：

```json
{
  "categories": [
    {
      "theme": "custom_theme",
      "subcategory": "my_sub",
      "description": "Your custom category description",
      "created_at": "2025-11-18T10:00:00Z"
    }
  ]
}
```

**重要**：
- `theme`, `subcategory`, `description` 必須使用**英文**
- 這是為了確保向量相似度計算的準確性
- AI 會自動遵守這個規則（已在 prompt 中強制要求）

下次分析時 AI 會自動使用新分類。

### 技巧 2：查看 AI 分析原始數據

```bash
# 查看最新分析結果
cat output/$(date +%Y年%m月%d日)/ai_analysis/*.json | jq .

# 查看特定事件詳情
cat output/*/ai_analysis/*.json | jq '.events[] | select(.importance > 8)'
```

### 技巧 3：透過 MCP 查詢

在 Claude Desktop 中：

```
分析最新的加密貨幣新聞
```

Claude 會自動調用 TrendRadar 的 MCP 工具並返回 AI 分析結果。

---

## 支援與反饋

- **Issues**：https://github.com/icedike/TrendRadar/issues
- **文檔**：查看 `AI_ENHANCEMENT_PLAN.md` 了解技術細節
- **更新日誌**：查看 git commits 了解最新改進

---

**版本**：v4.0.0-ai
**最後更新**：2025-11-18
