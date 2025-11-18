# TrendRadar 📡

> 智能加密貨幣新聞聚合與分析系統

TrendRadar 是一個強大的新聞監控工具，結合 RSS 抓取、AI 智能分析和 MCP 協議，為加密貨幣投資者提供全方位的資訊洞察。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AI](https://img.shields.io/badge/AI-Ollama-green.svg)](https://ollama.com)

---

## ✨ 核心功能

### 📰 多源新聞聚合
- 支援 NewsNow API 和多個 RSS 源
- 自動抓取加密貨幣相關新聞
- 關鍵字過濾與頻率統計
- 生成美觀的 HTML 報告

### 🤖 AI 智能分析（NEW!）
- **事件聚類**：自動合併相同事件的新聞，消除重複閱讀
- **主題分類**：智能分類到 10+ 主題（監管、市場、技術等）
- **重要性評分**：AI 評估事件重要程度（1-10 分）
- **自動摘要**：為每個事件生成簡潔摘要
- **情感分析**：分析新聞情感（正面/負面/中性）

**核心優勢**：
- ✅ 零 API 費用（使用本地 Ollama）
- ✅ 數據隱私保護（本地運行）
- ✅ 優雅降級（AI 失敗不影響主功能）

### 🔌 MCP 協議支援
- 支援 Claude Desktop 整合
- 13+ MCP 工具供 AI 助手調用
- 包含 AI 分析工具（查詢事件、分類、評分）

---

## 🚀 快速開始

### 1. 環境需求

- Python 3.8+
- （可選）Ollama - 用於 AI 分析

### 2. 安裝

```bash
# 克隆專案
git clone https://github.com/icedike/TrendRadar.git
cd TrendRadar

# 安裝依賴
pip install -r requirements.txt

# （可選）安裝 Ollama 以啟用 AI 功能
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# 下載推薦的 AI 模型
ollama pull llama3.2:3b
```

### 3. 配置

編輯 `config/config.yaml`：

```yaml
# 新聞源配置
rss_sources:
  - id: coindesk
    name: CoinDesk
    url: https://www.coindesk.com/arc/outboundfeeds/rss/
  # ... 更多源

# AI 分析配置（可選）
ai_analysis:
  enabled: true  # 啟用 AI 分析
  ollama_url: "http://127.0.0.1:11434"
  model: "llama3.2:3b"
  batch_size: 20
  cache_ttl_hours: 24
```

### 4. 運行

```bash
# 基礎模式（無 AI）
python main.py

# AI 增強模式
export AI_ANALYSIS_ENABLED=true
python main.py

# MCP 伺服器模式（供 Claude Desktop 使用）
python -m mcp_server.server
```

### 5. 查看結果

新聞報告會儲存在 `output/日期/` 目錄：
- `txt/` - 原始新聞數據
- `html/` - 美化的 HTML 報告
- `ai_analysis/` - AI 分析結果（JSON 格式）

---

## 📖 使用指南

### 傳統模式

```bash
python main.py
```

功能：
- 抓取最新新聞
- 關鍵字頻率統計
- 生成 HTML 報告
- 推送通知（可選）

### AI 增強模式

```bash
# 方式 1：環境變數
export AI_ANALYSIS_ENABLED=true
python main.py

# 方式 2：修改配置文件
# config.yaml 中設定 ai_analysis.enabled = true
python main.py
```

AI 功能：
- 自動聚類相同事件
- 智能分類新聞主題
- 評估重要性並排序
- 生成事件摘要
- 分析新聞情感

**詳細使用說明請查看 [AI_FEATURES.md](AI_FEATURES.md)**

### MCP 整合（Claude Desktop）

1. 配置 Claude Desktop 的 `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "trendradar": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/TrendRadar"
    }
  }
}
```

2. 重啟 Claude Desktop

3. 在對話中使用：
```
分析最新的加密貨幣新聞
查詢 Bitcoin 相關的重要事件
```

---

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────┐
│                    TrendRadar v4.0                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────────┐         ┌──────────────────┐   │
│  │   RSS 抓取     │────────>│  AI 分析         │   │
│  │   NewsAnalyzer │         │  AIAnalyzer      │   │
│  └────────────────┘         └──────────────────┘   │
│         │                            │              │
│         │                            ↓              │
│         │                   ┌──────────────────┐   │
│         │                   │  Ollama LLM      │   │
│         │                   │  (llama3.2:3b)   │   │
│         │                   └──────────────────┘   │
│         │                            │              │
│         ↓                            ↓              │
│  ┌──────────────────────────────────────────┐      │
│  │     output/日期/                          │      │
│  │     ├── txt/          (原始數據)          │      │
│  │     ├── ai_analysis/  (AI 結果)          │      │
│  │     └── html/         (增強報告)          │      │
│  └──────────────────────────────────────────┘      │
│                                                      │
│  ┌────────────────────────────────────────┐        │
│  │  MCP Server                             │        │
│  │  ├── 13+ 工具                           │        │
│  │  ├── AI 分析工具                        │        │
│  │  └── stdio/HTTP 傳輸                    │        │
│  └────────────────────────────────────────┘        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 🧪 測試

```bash
# 運行所有測試
python -m unittest discover tests -v

# 運行特定測試
python -m unittest tests.test_ai_analyzer -v
python -m unittest tests.test_vector_category_store -v
python -m unittest tests.test_batch_processing -v

# 測試統計
# - 34 個測試
# - 覆蓋 AI 分析、向量分類、批次處理等核心功能
```

---

## 📊 AI 功能詳解

### 事件聚類

**問題**：相同新聞在不同源重複出現

| 平台 | 標題 |
|------|------|
| CoinDesk | "Mt. Gox Moves $956M Worth of BTC..." |
| Cointelegraph | "Mt. Gox moves $953M Bitcoin..." |
| The Block | "Mt. Gox moves $956 million..." |

**解決方案**：AI 自動識別並合併為 1 個事件

```
事件：Mt. Gox 轉移大額比特幣
├─ 3 個來源
├─ 15 條新聞
├─ 重要性：8.5/10
└─ 摘要：Mt. Gox 錢包轉移約 $956M 比特幣...
```

### 主題分類

10 個預設主題 + AI 自動擴充：

- **regulation** - 監管政策
- **market** - 市場波動
- **technology** - 技術創新
- **defi** - 去中心化金融
- **nft** - NFT 與元宇宙
- **personnel** - 人事變動
- **security** - 安全事件
- **institutional** - 機構動態
- **macro** - 宏觀經濟
- **ecosystem** - 生態系統

### 重要性評分

AI 綜合考量：
- 報導數量（多源報導 = 更重要）
- 新聞排名（頭條 = 更重要）
- 市場影響（價格波動 = 更重要）
- 時效性（突發事件 = 更重要）

評分範圍：1-10 分（附帶信心度）

---

## 🎯 性能指標

| 指標 | 數值 | 說明 |
|------|------|------|
| 新聞抓取速度 | ~2 秒/源 | 取決於網路速度 |
| AI 分析速度 | 15-30 秒 | 50 篇新聞（llama3.2:3b） |
| 記憶體使用 | 2-4 GB | 包含 Ollama |
| 快取命中率 | >70% | 24 小時內相同新聞 |
| AI 聚類準確率 | ~85% | 實測數據 |
| AI 分類準確率 | ~90% | 實測數據 |

---

## 🔧 進階配置

### 批次處理

當新聞數量超過 `batch_size` 時自動啟用：

```yaml
ai_analysis:
  batch_size: 20  # 每批處理 20 篇新聞
```

優點：
- 避免 token 限制
- 提高處理速度
- 自動合併跨批次事件

### 向量分類庫

編輯 `output/category_store.json` 自定義分類：

```json
{
  "categories": [
    {
      "theme": "my_custom_theme",
      "subcategory": "my_sub",
      "description": "Custom category description"
    }
  ]
}
```

AI 會自動學習並使用新分類。

### 降級機制

AI 功能有完善的降級機制：

1. **Ollama 不可用** → 使用本地算法（標題相似度聚類、關鍵字分類）
2. **JSON 解析失敗** → 重試 3 次 → 降級
3. **任何錯誤** → 記錄日誌，不影響主流程

---

## 📚 文檔

- **[AI_FEATURES.md](AI_FEATURES.md)** - AI 功能完整使用指南
- **[AI_ENHANCEMENT_PLAN.md](AI_ENHANCEMENT_PLAN.md)** - AI 功能技術設計文檔
- **[CLAUDE.md](CLAUDE.md)** - MCP 開發指南（如果存在）

---

## 🤝 貢獻

歡迎貢獻！請遵循以下步驟：

1. Fork 專案
2. 創建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交改動 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

### 開發指南

```bash
# 安裝開發依賴
pip install -r requirements.txt

# 運行測試
python -m unittest discover tests -v

# 運行 linter（如果有）
flake8 *.py mcp_server/

# 檢查代碼覆蓋率（如果有 coverage）
coverage run -m unittest discover
coverage report
```

---

## ⚠️ 注意事項

1. **API 限制**：部分新聞源可能有請求頻率限制
2. **AI 準確率**：AI 分析結果僅供參考，建議結合人工判斷
3. **記憶體需求**：啟用 AI 功能需要額外 2-5GB RAM
4. **Ollama 依賴**：AI 功能需要 Ollama 運行，否則自動降級

---

## 🛠️ 故障排除

### AI 分析失敗

```bash
# 檢查 Ollama 是否運行
ollama list

# 重啟 Ollama
ollama serve

# 測試模型
ollama run llama3.2:3b "Hello"
```

### 記憶體不足

```bash
# 使用更小的模型
ollama pull llama3.2:3b  # 僅需 2GB

# 減少批次大小
# config.yaml: batch_size: 10
```

### 更多問題

查看 [AI_FEATURES.md](AI_FEATURES.md#故障排除) 的詳細故障排除章節。

---

## 📄 授權

本專案採用 MIT 授權 - 詳見 [LICENSE](LICENSE) 文件。

---

## 👥 作者

- **icedike** - *初始工作* - [GitHub](https://github.com/icedike)

---

## 🙏 致謝

- [Ollama](https://ollama.com) - 本地 LLM 運行環境
- [Anthropic](https://anthropic.com) - MCP 協議
- 所有新聞源提供者

---

## 📮 聯繫

- GitHub Issues: https://github.com/icedike/TrendRadar/issues
- 專案連結: https://github.com/icedike/TrendRadar

---

**版本**：v4.0.0-ai
**最後更新**：2025-11-18

⭐ 如果這個專案對你有幫助，請給個 Star！
