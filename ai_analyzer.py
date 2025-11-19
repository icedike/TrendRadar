"""AI-powered news analysis pipeline for TrendRadar."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson
import pytz
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from ai_prompts import (
    CLASSIFICATION_PROMPT,
    CLUSTER_PROMPT_TEMPLATE,
    SCORING_PROMPT,
    SUMMARY_PROMPT,
)
from vector_category_store import VectorCategoryStore


BEIJING_TZ = pytz.timezone("Asia/Shanghai")


def _now() -> datetime:
    return datetime.now(BEIJING_TZ)


@dataclass
class AIAnalyzerConfig:
    enabled: bool = False
    model: str = "llama3.2:3b"
    ollama_url: str = "http://127.0.0.1:11434"
    batch_size: int = 20
    cache_ttl_hours: int = 24
    output_version: str = "1.0"
    output_root: str = "output"


class OllamaClientError(RuntimeError):
    """Raised when the Ollama client cannot complete a request."""


class OllamaClient:
    """Thin wrapper around the Ollama python SDK with retry logic."""

    def __init__(self, base_url: str, model: str, enabled: bool, max_retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self.max_retries = max_retries
        self.session = requests.Session() if self.enabled else None

    def is_available(self) -> bool:
        return self.enabled and self.session is not None

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_available():
            raise OllamaClientError("Ollama client not available")
        url = f"{self.base_url}{path}"
        response = self.session.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def generate(self, prompt: str, format: str = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if format:
            payload["format"] = format
        response = self._post("/api/generate", payload)
        content = response.get("response") or response.get("message")
        if not content:
            raise OllamaClientError("Empty response from Ollama")
        return content

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def chat(self, messages: List[Dict[str, str]], format: str = None) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        if format:
            payload["format"] = format
        response = self._post("/api/chat", payload)
        message = response.get("message", {})
        content = message.get("content")
        if not content:
            raise OllamaClientError("Empty response from Ollama chat")
        return content


class AIResultRepository:
    """Persist AI analysis outputs under output/<date>/ai_analysis."""

    def __init__(
        self,
        output_root: str = "output",
        cache_ttl_hours: int = 24,
    ) -> None:
        self.output_root = Path(output_root)
        self.cache_ttl = timedelta(hours=max(cache_ttl_hours, 1))

    def _date_folder(self, date: Optional[datetime] = None) -> str:
        if isinstance(date, datetime):
            target = date.astimezone(BEIJING_TZ)
        else:
            target = _now() if date is None else date
            if not isinstance(target, datetime):
                raise ValueError("date must be datetime or None")
            target = target.astimezone(BEIJING_TZ)
        return target.strftime("%Y年%m月%d日")

    def _dir_for_date(self, date: Optional[datetime] = None) -> Path:
        folder = self._date_folder(date)
        ai_dir = self.output_root / folder / "ai_analysis"
        ai_dir.mkdir(parents=True, exist_ok=True)
        return ai_dir

    def _cache_path(self, date: Optional[datetime] = None) -> Path:
        return self._dir_for_date(date) / "cache.json"

    def _load_cache(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        cache_file = self._cache_path(date)
        if not cache_file.exists():
            return {"entries": {}}
        try:
            return orjson.loads(cache_file.read_bytes())
        except Exception as e:
            print(f"⚠️  [AIResultRepository] 載入快取失敗 ({cache_file}): {type(e).__name__}: {e}")
            return {"entries": {}}

    def _save_cache(self, cache: Dict[str, Any], date: Optional[datetime] = None) -> None:
        cache_file = self._cache_path(date)
        cache_file.write_bytes(
            orjson.dumps(cache, option=orjson.OPT_INDENT_2)
        )

    def load_latest(self, date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        ai_dir = self._dir_for_date(date)
        json_files = sorted(
            [f for f in ai_dir.glob("*.json") if f.name != "cache.json"],
            key=lambda p: p.stat().st_mtime,
        )
        if not json_files:
            return None
        latest = json_files[-1]
        try:
            return orjson.loads(latest.read_bytes())
        except Exception as e:
            print(f"⚠️  [AIResultRepository] 載入分析結果失敗 ({latest}): {type(e).__name__}: {e}")
            return None

    def _is_entry_valid(self, entry: Dict[str, Any]) -> bool:
        timestamp = entry.get("generated_at")
        if not timestamp:
            return False
        try:
            generated_time = datetime.fromisoformat(timestamp)
        except ValueError:
            return False
        if generated_time.tzinfo is None:
            generated_time = generated_time.replace(tzinfo=BEIJING_TZ)
        return _now() - generated_time <= self.cache_ttl

    def get_cache_entry(self, payload_hash: str, date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        cache = self._load_cache(date)
        entry = cache.get("entries", {}).get(payload_hash)
        if not entry or not self._is_entry_valid(entry):
            return None
        ai_dir = self._dir_for_date(date)
        file_path = ai_dir / entry.get("file_name", "")
        if not file_path.exists():
            return None
        try:
            data = orjson.loads(file_path.read_bytes())
            data.setdefault("ai_status", "cached")
            return data
        except Exception as e:
            print(f"⚠️  [AIResultRepository] 從快取讀取分析結果失敗 ({file_path}): {type(e).__name__}: {e}")
            return None

    def put_cache_entry(
        self,
        payload_hash: str,
        file_name: str,
        generated_at: str,
        date: Optional[datetime] = None,
    ) -> None:
        cache = self._load_cache(date)
        cache.setdefault("entries", {})[payload_hash] = {
            "file_name": file_name,
            "generated_at": generated_at,
        }
        self._save_cache(cache, date)

    def save(self, result: Dict[str, Any], date: Optional[datetime] = None) -> Dict[str, Any]:
        ai_dir = self._dir_for_date(date)
        timestamp = result.get("generated_at") or _now().isoformat()
        if "generated_at" not in result:
            result["generated_at"] = timestamp
        file_name = result.get("file_name")
        if not file_name:
            slug = timestamp.replace(":", "-").replace(" ", "_")
            file_name = f"{slug}-ai-analysis.json"
            result["file_name"] = file_name
        file_path = ai_dir / file_name
        file_path.write_bytes(
            orjson.dumps(result, option=orjson.OPT_INDENT_2)
        )
        if result.get("payload_hash"):
            self.put_cache_entry(result["payload_hash"], file_name, timestamp, date)
        return result


class AIAnalyzer:
    """Coordinator for the AI-enhanced news pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        platform_configs: List[Dict[str, str]],
        repository: Optional[AIResultRepository] = None,
    ) -> None:
        self.config = AIAnalyzerConfig(
            enabled=config.get("ENABLED", False),
            model=config.get("OLLAMA_MODEL", "llama3.2:3b"),
            ollama_url=config.get("OLLAMA_URL", "http://127.0.0.1:11434"),
            batch_size=int(config.get("BATCH_SIZE", 20)),
            cache_ttl_hours=int(config.get("CACHE_TTL_HOURS", 24)),
            output_version=config.get("OUTPUT_VERSION", "1.0"),
            output_root=config.get("OUTPUT_ROOT", "output"),
        )
        self.platform_name_map = {
            platform.get("id"): platform.get("name", platform.get("id"))
            for platform in platform_configs
            if platform.get("id")
        }
        self.repository = repository or AIResultRepository(
            output_root=self.config.output_root,
            cache_ttl_hours=self.config.cache_ttl_hours,
        )
        category_path = self.repository.output_root / "category_store.json"
        self.category_store = VectorCategoryStore(category_path)
        self.ollama_client = OllamaClient(
            base_url=self.config.ollama_url,
            model=self.config.model,
            enabled=self.config.enabled,
        )

    def analyze(
        self,
        raw_results: Dict[str, Dict[str, Dict[str, Any]]],
        title_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not raw_results:
            return None

        payload = self.prepare_ai_payload(raw_results, title_info)
        if not payload:
            return None

        payload_hash = self._hash_payload(payload)
        cached = self.repository.get_cache_entry(payload_hash)
        if cached:
            cached.setdefault("ai_status", "cached")
            return cached

        events = self.cluster_events(payload)
        enriched_events = []
        for event in events:
            classification = self.classify_theme(event)
            importance = self.score_importance(event)
            sentiment = self.analyze_sentiment(event)
            summary = self.generate_summary(event)
            event.update(classification)
            event.update(importance)
            event.update(sentiment)
            event["summary"] = summary
            enriched_events.append(event)

        result = self._build_result(enriched_events, payload, payload_hash)
        saved = self.repository.save(result)
        saved.setdefault("ai_status", "ok")
        return saved

    def prepare_ai_payload(
        self,
        raw_results: Dict[str, Dict[str, Dict[str, Any]]],
        title_info: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        articles: List[Dict[str, Any]] = []
        for platform_id, titles in raw_results.items():
            platform_name = self.platform_name_map.get(platform_id, platform_id)
            for idx, (title, info) in enumerate(titles.items(), start=1):
                article_id = f"{platform_id}:{idx}:{hashlib.md5(title.encode('utf-8')).hexdigest()[:6]}"
                ranks = info.get("ranks", [])
                timestamp = ""
                if (
                    title_info
                    and platform_id in title_info
                    and title in title_info[platform_id]
                ):
                    timestamp = title_info[platform_id][title].get("last_time", "")
                articles.append(
                    {
                        "article_id": article_id,
                        "platform_id": platform_id,
                        "platform_name": platform_name,
                        "title": title,
                        "url": info.get("url", ""),
                        "mobile_url": info.get("mobileUrl", ""),
                        "ranks": ranks,
                        "source_rank": ranks[0] if ranks else None,
                        "timestamp": timestamp,
                    }
                )
        return articles

    def cluster_events(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not articles:
            return []

        # If article count exceeds batch size, use batch processing
        if len(articles) > self.config.batch_size and self.ollama_client.is_available():
            print(f"ℹ️  [cluster_events] 處理 {len(articles)} 篇文章，使用批次處理（批次大小: {self.config.batch_size}）")
            return self._cluster_events_batched(articles)

        # For smaller datasets, process all at once
        if self.ollama_client.is_available():
            prompt = CLUSTER_PROMPT_TEMPLATE.format(
                payload=json.dumps(articles, ensure_ascii=False)
            )
            # Retry up to 3 times for JSON parsing errors
            for attempt in range(3):
                try:
                    response = self.ollama_client.generate(prompt, format="json")
                    parsed = json.loads(response)
                    events = parsed.get("events", [])
                    return self._merge_events_with_articles(events, articles)
                except json.JSONDecodeError as e:
                    print(f"⚠️  [cluster_events] LLM 返回無效 JSON (嘗試 {attempt + 1}/3): {e}")
                    if attempt == 2:  # Last attempt
                        print("❌ [cluster_events] JSON 解析失敗 3 次，降級到本地聚類")
                        break
                    # Wait before retry (exponential backoff: 1s, 2s)
                    time.sleep(1 * (attempt + 1))
                except Exception as e:
                    print(f"❌ [cluster_events] 聚類過程發生意外錯誤: {type(e).__name__}: {e}")
                    break
        return self._cluster_locally(articles)

    def _cluster_events_batched(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster articles in batches to avoid token limits."""
        batch_size = self.config.batch_size
        all_events: List[Dict[str, Any]] = []

        # Process articles in batches
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            print(f"ℹ️  [cluster_events] 處理批次 {i // batch_size + 1}/{(len(articles) + batch_size - 1) // batch_size}")

            prompt = CLUSTER_PROMPT_TEMPLATE.format(
                payload=json.dumps(batch, ensure_ascii=False)
            )

            # Retry up to 3 times for JSON parsing errors
            batch_events = None
            for attempt in range(3):
                try:
                    response = self.ollama_client.generate(prompt, format="json")
                    parsed = json.loads(response)
                    events = parsed.get("events", [])
                    batch_events = self._merge_events_with_articles(events, batch)
                    break
                except json.JSONDecodeError as e:
                    print(f"⚠️  [cluster_events_batched] 批次 {i // batch_size + 1} LLM 返回無效 JSON (嘗試 {attempt + 1}/3): {e}")
                    if attempt == 2:  # Last attempt
                        print(f"❌ [cluster_events_batched] 批次 {i // batch_size + 1} JSON 解析失敗，使用本地聚類")
                        batch_events = self._cluster_locally(batch)
                        break
                    time.sleep(1 * (attempt + 1))
                except Exception as e:
                    print(f"❌ [cluster_events_batched] 批次 {i // batch_size + 1} 發生錯誤: {type(e).__name__}: {e}")
                    batch_events = self._cluster_locally(batch)
                    break

            if batch_events:
                all_events.extend(batch_events)

        # Merge similar events across batches
        return self._merge_cross_batch_events(all_events)

    def _merge_events_with_articles(
        self, events: List[Dict[str, Any]], articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        article_map = {article["article_id"]: article for article in articles}
        merged = []
        for event in events:
            refs = event.get("article_refs") or []
            event_articles = [article_map[a_id] for a_id in refs if a_id in article_map]
            if not event_articles:
                continue
            merged.append(
                {
                    "event_id": event.get("event_id")
                    or f"event_{hashlib.md5(event_articles[0]['title'].encode('utf-8')).hexdigest()[:8]}",
                    "title": event.get("title") or event_articles[0]["title"],
                    "articles": event_articles,
                    "rationale": event.get("rationale", ""),
                }
            )
        return merged

    def _cluster_locally(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clusters: Dict[str, Dict[str, Any]] = {}
        for article in articles:
            key = self._normalize(article["title"])
            if key not in clusters:
                clusters[key] = {
                    "event_id": f"event_{hashlib.md5(key.encode('utf-8')).hexdigest()[:8]}",
                    "title": article["title"],
                    "articles": [],
                    "rationale": "Grouped by normalized title similarity",
                }
            clusters[key]["articles"].append(article)
        return list(clusters.values())

    def classify_theme(self, event: Dict[str, Any]) -> Dict[str, str]:
        title = event.get("title", "")
        context = self._compose_event_context(event)
        vector_hit = self.category_store.lookup(title, context)
        if vector_hit:
            return {
                "theme": vector_hit["theme"],
                "subcategory": vector_hit["subcategory"],
                "classification_source": "vector_store",
                "similarity": vector_hit.get("similarity"),
            }
        if self.ollama_client.is_available():
            prompt = CLASSIFICATION_PROMPT.format(
                event_summary=context or title
            )
            # Retry up to 3 times for JSON parsing errors
            for attempt in range(3):
                try:
                    response = self.ollama_client.generate(prompt, format="json")
                    data = json.loads(response)
                    if "theme" in data:
                        theme = self._normalize_category(data.get("theme", "general"))
                        subcategory = self._normalize_category(
                            data.get("subcategory", "overview")
                        )
                        explanation = data.get("explanation") or title
                        self.category_store.record_classification(
                            theme,
                            subcategory,
                            explanation,
                            source_event=title,
                        )
                        return {
                            "theme": theme,
                            "subcategory": subcategory,
                            "classification_source": "llm",
                        }
                except json.JSONDecodeError as e:
                    print(f"⚠️  [classify_theme] LLM 返回無效 JSON (嘗試 {attempt + 1}/3): {e}")
                    if attempt == 2:  # Last attempt
                        print("❌ [classify_theme] JSON 解析失敗 3 次，降級到關鍵字分類")
                        break
                    time.sleep(1 * (attempt + 1))
                except Exception as e:
                    print(f"❌ [classify_theme] 分類過程發生意外錯誤: {type(e).__name__}: {e}")
                    break

        title_lower = title.lower()
        keywords = {
            "regulation": ["sec", "regulation", "law", "policy"],
            "market": ["price", "market", "etf", "trading", "bull", "bear"],
            "technology": ["upgrade", "fork", "release", "layer", "tech"],
            "defi": ["defi", "dex", "liquidity", "staking"],
            "nft": ["nft", "collectible", "metaverse"],
            "personnel": ["ceo", "hire", "resign", "founder"],
            "security": ["hack", "exploit", "breach", "attack"],
            "institutional": ["blackrock", "goldman", "bank", "institution"],
            "macro": ["inflation", "fed", "economy", "jobs"],
            "ecosystem": ["community", "ecosystem", "partnership"],
        }
        for theme, words in keywords.items():
            if any(word in title_lower for word in words):
                return {
                    "theme": theme,
                    "subcategory": words[0],
                    "classification_source": "keyword",
                }
        return {"theme": "general", "subcategory": "overview"}

    def score_importance(self, event: Dict[str, Any]) -> Dict[str, float]:
        articles = event.get("articles", [])
        context = "\n".join(article["title"] for article in articles[:5])
        if self.ollama_client.is_available() and context:
            prompt = SCORING_PROMPT.format(event_context=context)
            # Retry up to 3 times for JSON parsing errors
            for attempt in range(3):
                try:
                    response = self.ollama_client.generate(prompt, format="json")
                    data = json.loads(response)
                    importance = float(data.get("importance", 0))
                    confidence = float(data.get("confidence", 0.5))

                    # Validate for NaN/Infinity
                    if not (math.isfinite(importance) and math.isfinite(confidence)):
                        raise ValueError(f"Invalid numeric values from LLM: importance={importance}, confidence={confidence}")

                    return {
                        "importance": max(1.0, min(10.0, round(importance, 2))),
                        "confidence": max(0.0, min(0.99, round(confidence, 2))),
                    }
                except json.JSONDecodeError as e:
                    print(f"⚠️  [score_importance] LLM 返回無效 JSON (嘗試 {attempt + 1}/3): {e}")
                    if attempt == 2:  # Last attempt
                        print("❌ [score_importance] JSON 解析失敗 3 次，降級到啟發式評分")
                        break
                    time.sleep(1 * (attempt + 1))
                except (ValueError, KeyError) as e:
                    print(f"⚠️  [score_importance] 數據格式錯誤: {type(e).__name__}: {e}")
                    break
                except Exception as e:
                    print(f"❌ [score_importance] 評分過程發生意外錯誤: {type(e).__name__}: {e}")
                    break
        # Heuristic fallback scoring
        count_score = min(len(articles) / 3, 1)
        ranks = [article.get("source_rank") for article in articles if article.get("source_rank")]
        rank_score = 0
        if ranks:
            best_rank = min(ranks)
            rank_score = max(0, (10 - min(best_rank, 10)) / 10)
        importance = max(1.0, min(10.0, (count_score * 6 + rank_score * 4) + 2))
        confidence = round(0.5 + 0.05 * len(articles), 2)
        return {"importance": round(importance, 2), "confidence": min(confidence, 0.95)}

    def analyze_sentiment(self, event: Dict[str, Any]) -> Dict[str, str]:
        title = event.get("title", "").lower()
        negative = ["hack", "exploit", "down", "crash", "probe"]
        positive = ["record", "approval", "launch", "surge", "gain"]
        if any(word in title for word in negative):
            return {"sentiment": "negative"}
        if any(word in title for word in positive):
            return {"sentiment": "positive"}
        return {"sentiment": "neutral"}

    def generate_summary(self, event: Dict[str, Any]) -> str:
        articles = event.get("articles", [])
        snippets = [article["title"] for article in articles[:3]]
        context = "; ".join(snippets)
        if self.ollama_client.is_available():
            prompt = SUMMARY_PROMPT.format(event_context=context)
            # Retry up to 2 times for summary generation
            for attempt in range(2):
                try:
                    # Note: Do not pass format="json" here; summary is expected as plain text.
                    summary = self.ollama_client.generate(prompt)
                    # Validate summary is not empty and reasonable length
                    if summary and len(summary.strip()) > 10:
                        return summary.strip()
                    print(f"⚠️  [generate_summary] LLM 返回空或過短的摘要 (嘗試 {attempt + 1}/2)")
                except Exception as e:
                    print(f"⚠️  [generate_summary] 摘要生成失敗 (嘗試 {attempt + 1}/2): {type(e).__name__}: {e}")
                if attempt < 1:  # Wait before retry
                    time.sleep(1)
            print("❌ [generate_summary] 摘要生成失敗，使用截斷的上下文")
        return context[:240]

    def _build_result(
        self,
        events: List[Dict[str, Any]],
        payload: List[Dict[str, Any]],
        payload_hash: str,
    ) -> Dict[str, Any]:
        sources: Dict[str, Dict[str, Any]] = {}
        for article in payload:
            platform_id = article["platform_id"]
            sources.setdefault(
                platform_id,
                {
                    "platform_name": article.get("platform_name", platform_id),
                    "articles": [],
                },
            )
            sources[platform_id]["articles"].append(article)
        return {
            "version": self.config.output_version,
            "model": self.config.model,
            "generated_at": _now().isoformat(),
            "payload_hash": payload_hash,
            "ai_status": "ok",
            "events": events,
            "sources": sources,
            "total_articles": len(payload),
        }

    def _hash_payload(self, payload: List[Dict[str, Any]]) -> str:
        serialized = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
        return hashlib.md5(serialized).hexdigest()

    def _compose_event_context(self, event: Dict[str, Any]) -> str:
        parts: List[str] = []
        title = event.get("title")
        if title:
            parts.append(title)
        rationale = event.get("rationale")
        if rationale:
            parts.append(rationale)
        articles = event.get("articles") or []
        if articles:
            snippets = [article.get("title", "") for article in articles[:3]]
            snippets = [snippet for snippet in snippets if snippet]
            if snippets:
                parts.append("; ".join(snippets))
        return " | ".join(parts)

    def _merge_cross_batch_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge similar events that may have been split across batches."""
        if not events:
            return []

        # Group events by normalized title
        title_groups: Dict[str, List[Dict[str, Any]]] = {}
        for event in events:
            normalized = self._normalize(event.get("title", ""))
            if normalized not in title_groups:
                title_groups[normalized] = []
            title_groups[normalized].append(event)

        # Merge grouped events
        merged_events = []
        for group in title_groups.values():
            if len(group) == 1:
                merged_events.append(group[0])
            else:
                # Merge multiple events with same normalized title
                merged = {
                    "event_id": group[0]["event_id"],
                    "title": group[0]["title"],
                    "articles": [],
                    "rationale": group[0].get("rationale", "")
                }
                for event in group:
                    merged["articles"].extend(event.get("articles", []))
                merged_events.append(merged)

        return merged_events

    def _normalize_category(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
        return slug or "general"

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    def cluster_and_transform_data(
        self,
        raw_results: Dict[str, Dict[str, Dict[str, Any]]],
        title_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], bool]:
        """
        将基于标题的数据结构转换为基于事件的数据结构

        输入: raw_results = {platform_id: {title: {ranks, url, ...}}}
        输出: event_results = {platform_id: {event_id: {event_title, articles, frequency, importance, ...}}}
        返回: (event_results, has_ai_scores)
        """
        if not raw_results:
            return {}, False

        # 准备 AI payload
        payload = self.prepare_ai_payload(raw_results, title_info)
        if not payload:
            return self._fallback_title_clustering(raw_results, title_info), False  # 降级：返回原始数据

        # 检查是否启用 AI
        if not self.ollama_client.is_available():
            print("⚠️  AI 不可用，使用标题归一化降级模式")
            return self._fallback_title_clustering(raw_results, title_info), False

        # 执行 AI 聚类
        try:
            events = self.cluster_events(payload)
            if not events:
                print("⚠️  AI 聚类返回空结果，使用降级模式")
                return self._fallback_title_clustering(raw_results, title_info), False

            # 为每个事件添加分类、评分等
            enriched_events = []
            for event in events:
                classification = self.classify_theme(event)
                importance = self.score_importance(event)
                sentiment = self.analyze_sentiment(event)
                summary = self.generate_summary(event)
                event.update(classification)
                event.update(importance)
                event.update(sentiment)
                event["summary"] = summary
                enriched_events.append(event)

            # 转换为新的数据结构
            event_results = self._transform_events_to_dict(enriched_events, title_info)
            return event_results, True

        except Exception as e:
            print(f"❌ AI 聚类失败: {e}，使用降级模式")
            return self._fallback_title_clustering(raw_results, title_info), False

    def _transform_events_to_dict(
        self,
        events: List[Dict[str, Any]],
        title_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        将事件列表转换为字典结构
        {platform_id: {event_id: event_data}}
        """
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for event in events:
            event_id = event.get("event_id", "unknown")
            articles = event.get("articles", [])

            if not articles:
                continue

            # 按平台分组文章
            platform_articles: Dict[str, List[Dict]] = {}
            for article in articles:
                platform_id = article.get("platform_id", "unknown")
                if platform_id not in platform_articles:
                    platform_articles[platform_id] = []
                platform_articles[platform_id].append(article)

            # 为每个平台创建事件条目
            for platform_id, plat_articles in platform_articles.items():
                if platform_id not in result:
                    result[platform_id] = {}

                # 收集所有排名
                all_ranks = []
                first_time = None
                last_time = None
                urls = []

                for article in plat_articles:
                    ranks = article.get("ranks", [])
                    all_ranks.extend(ranks)

                    # URL
                    url = article.get("url", "")
                    if url and url not in urls:
                        urls.append(url)

                    # 时间信息
                    timestamp = article.get("timestamp", "")
                    if timestamp:
                        if first_time is None or timestamp < first_time:
                            first_time = timestamp
                        if last_time is None or timestamp > last_time:
                            last_time = timestamp

                # 创建事件数据
                event_data = {
                    "event_title": event.get("title", ""),
                    "articles": plat_articles,  # 包含原始文章列表
                    "frequency": len(plat_articles),  # 事件频率 = 文章数
                    "ranks": all_ranks if all_ranks else [99],
                    "url": urls[0] if urls else "",
                    "urls": urls,  # 所有URL
                    "mobileUrl": plat_articles[0].get("mobile_url", "") if plat_articles else "",
                    "first_time": first_time or "",
                    "last_time": last_time or "",
                    "importance": event.get("importance", 5.0),
                    "confidence": event.get("confidence", 0.5),
                    "theme": event.get("theme", "general"),
                    "subcategory": event.get("subcategory", ""),
                    "sentiment": event.get("sentiment", "neutral"),
                    "summary": event.get("summary", ""),
                    "has_ai_score": True,
                }

                result[platform_id][event_id] = event_data

        return result

    def _fallback_title_clustering(
        self,
        raw_results: Dict[str, Dict[str, Dict[str, Any]]],
        title_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        降级模式：基于标题归一化的简单聚类
        将相似标题（去除标点、大小写后相同）归为一个事件
        
        注意：此函数要求 raw_results 为标题字典格式 {platform_id: {title: {data}}}
        而不是事件字典格式。title_info 也应为标题索引的时间信息。
        """
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for platform_id, titles_data in raw_results.items():
            result[platform_id] = {}

            # 按归一化标题分组
            normalized_groups: Dict[str, List[Tuple[str, Dict]]] = {}
            for title, data in titles_data.items():
                normalized = self._normalize(title)
                if normalized not in normalized_groups:
                    normalized_groups[normalized] = []
                normalized_groups[normalized].append((title, data))

            # 为每组创建事件
            for group_idx, (normalized, title_list) in enumerate(normalized_groups.items()):
                # 使用第一个标题作为事件标题
                first_title = title_list[0][0]
                event_id = f"event_{platform_id}_{group_idx}_{normalized[:8]}"

                # 合并所有数据
                all_ranks = []
                urls = []
                for title, data in title_list:
                    ranks = data.get("ranks", [])
                    all_ranks.extend(ranks)
                    url = data.get("url", "")
                    if url and url not in urls:
                        urls.append(url)

                # 获取时间信息
                first_time = ""
                last_time = ""
                if title_info and platform_id in title_info:
                    for title, _ in title_list:
                        if title in title_info[platform_id]:
                            info = title_info[platform_id][title]
                            ft = info.get("first_time", "")
                            lt = info.get("last_time", "")
                            if ft:
                                if not first_time or ft < first_time:
                                    first_time = ft
                            if lt:
                                if not last_time or lt > last_time:
                                    last_time = lt

                # 创建简化的事件数据（没有AI评分）
                # 构建与 AI 模式兼容的 articles 列表
                articles_list = []
                platform_name = self.platform_name_map.get(platform_id, platform_id)
                for idx, (title, data) in enumerate(title_list, start=1):
                    article_id = f"{platform_id}:{idx}:{hashlib.md5(title.encode('utf-8')).hexdigest()[:6]}"
                    timestamp = ""
                    if title_info and platform_id in title_info and title in title_info[platform_id]:
                        timestamp = title_info[platform_id][title].get("last_time", "")

                    articles_list.append({
                        "article_id": article_id,
                        "platform_id": platform_id,
                        "platform_name": platform_name,
                        "title": title,
                        "url": data.get("url", ""),
                        "mobile_url": data.get("mobileUrl", ""),
                        "ranks": data.get("ranks", []),
                        "source_rank": data.get("ranks", [None])[0] if data.get("ranks") else None,
                        "timestamp": timestamp,
                    })

                event_data = {
                    "event_title": first_title,
                    "articles": articles_list,
                    "frequency": len(title_list),
                    "ranks": all_ranks if all_ranks else [99],
                    "url": urls[0] if urls else "",
                    "urls": urls,
                    "mobileUrl": title_list[0][1].get("mobileUrl", ""),
                    "first_time": first_time,
                    "last_time": last_time,
                    "has_ai_score": False,  # 标记为降级模式
                }

                result[platform_id][event_id] = event_data

        return result


__all__ = ["AIAnalyzer", "AIResultRepository", "AIAnalyzerConfig", "OllamaClient", "OllamaClientError"]
