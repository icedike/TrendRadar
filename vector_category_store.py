"""Lightweight vector-based category store for AI classification."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CATEGORY_DEFINITIONS: List[Dict[str, str]] = [
    {"theme": "regulation", "subcategory": "policy", "description": "Global regulatory actions, compliance policies, SEC/CFTC statements."},
    {"theme": "market", "subcategory": "price_action", "description": "Market structure, ETFs, price swings, liquidity and trading volume."},
    {"theme": "technology", "subcategory": "infrastructure", "description": "Protocol upgrades, client releases, scalability breakthroughs, layer-2 updates."},
    {"theme": "defi", "subcategory": "liquidity", "description": "Decentralized exchanges, liquidity pools, staking and yield mechanics."},
    {"theme": "nft", "subcategory": "collectibles", "description": "NFT launches, gaming integrations, metaverse partnerships."},
    {"theme": "personnel", "subcategory": "leadership", "description": "Executive moves, hires, resignations and governance appointments."},
    {"theme": "security", "subcategory": "exploit", "description": "Hacks, exploits, security disclosures, chain halts."},
    {"theme": "institutional", "subcategory": "adoption", "description": "Banks, asset managers, corporates adopting crypto services."},
    {"theme": "macro", "subcategory": "economy", "description": "Inflation, interest rates, GDP, macroeconomic indicators influencing crypto."},
    {"theme": "ecosystem", "subcategory": "community", "description": "Partnerships, grants, developer or community ecosystem moves."},
]


@dataclass
class CategoryRecord:
    theme: str
    subcategory: str
    description: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    source_event: str = ""


class VectorCategoryStore:
    """Tiny cosine-similarity store backed by JSON for deterministic category hits."""

    def __init__(
        self,
        storage_path: Path,
        similarity_threshold: float = 0.82,
        base_definitions: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.similarity_threshold = similarity_threshold
        self.base_definitions = base_definitions or DEFAULT_CATEGORY_DEFINITIONS
        self.categories: List[Dict[str, Any]] = []
        self._load()
        if not self.categories:
            self._seed_defaults()
            self._save()

    def _seed_defaults(self) -> None:
        for item in self.base_definitions:
            record = CategoryRecord(
                theme=item["theme"],
                subcategory=item["subcategory"],
                description=item["description"],
            )
            self.categories.append(record.__dict__)

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "categories" in payload:
                self.categories = payload.get("categories", [])
            elif isinstance(payload, list):
                self.categories = payload
        except Exception:
            self.categories = []

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            {k: v for k, v in entry.items() if k != "_vector"}
            for entry in self.categories
        ]
        data = {"vector_version": "v1", "categories": serializable}
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _embed(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = float(sum(counts.values()))
        norm = math.sqrt(sum((c / total) ** 2 for c in counts.values())) or 1.0
        return {token: (count / total) / norm for token, count in counts.items()}

    @staticmethod
    def _cosine(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        return sum(weight * vec_b.get(token, 0.0) for token, weight in vec_a.items())

    def lookup(self, title: str, context: str = "") -> Optional[Dict[str, Any]]:
        query = " ".join(part for part in [title or "", context or ""] if part).strip()
        if not query:
            return None
        query_vec = self._embed(query)
        if not query_vec or not self.categories:
            return None
        best_entry: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for entry in self.categories:
            vector = entry.get("_vector")
            if vector is None:
                vector = self._embed(entry.get("description") or f"{entry['theme']} {entry['subcategory']}")
                entry["_vector"] = vector
            score = self._cosine(query_vec, vector)
            if score > best_score:
                best_entry = entry
                best_score = score
        if best_entry and best_score >= self.similarity_threshold:
            return {
                "theme": best_entry.get("theme", "general"),
                "subcategory": best_entry.get("subcategory", "overview"),
                "description": best_entry.get("description", ""),
                "similarity": round(best_score, 4),
            }
        return None

    def record_classification(
        self,
        theme: str,
        subcategory: str,
        description: str,
        source_event: str = "",
    ) -> None:
        theme = (theme or "general").strip().lower().replace(" ", "_")
        subcategory = (subcategory or "overview").strip().lower().replace(" ", "_")
        description = description.strip() if description else f"{theme}:{subcategory}"
        timestamp = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        for entry in self.categories:
            if entry.get("theme") == theme and entry.get("subcategory") == subcategory:
                entry["description"] = description
                entry["source_event"] = source_event or entry.get("source_event", "")
                entry["updated_at"] = timestamp
                entry.pop("_vector", None)
                self._save()
                return
        record = CategoryRecord(
            theme=theme,
            subcategory=subcategory,
            description=description,
            source_event=source_event,
        ).__dict__
        self.categories.append(record)
        self._save()
