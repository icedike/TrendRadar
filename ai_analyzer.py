"""AI-powered news analysis pipeline for TrendRadar."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        except Exception:
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
        except Exception:
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
        except Exception:
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

    def _normalize_category(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
        return slug or "general"

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())


__all__ = ["AIAnalyzer", "AIResultRepository", "AIAnalyzerConfig", "OllamaClient", "OllamaClientError"]
