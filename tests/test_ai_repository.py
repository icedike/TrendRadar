import tempfile
import unittest
from datetime import datetime, timedelta

from ai_analyzer import AIResultRepository


class AIResultRepositoryTest(unittest.TestCase):
    def test_save_and_retrieve_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = AIResultRepository(output_root=tmp, cache_ttl_hours=24)
            payload = {
                "version": "1.0",
                "model": "test",
                "generated_at": datetime.now().isoformat(),
                "payload_hash": "hash123",
                "events": [],
                "sources": {},
                "total_articles": 0,
            }
            repo.save(payload)

            latest = repo.load_latest()
            self.assertIsNotNone(latest)
            self.assertEqual(latest["payload_hash"], "hash123")

            cached = repo.get_cache_entry("hash123")
            self.assertIsNotNone(cached)
            self.assertEqual(cached["payload_hash"], "hash123")

    def test_cache_expiration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = AIResultRepository(output_root=tmp, cache_ttl_hours=1)
            old_time = (datetime.now() - timedelta(hours=5)).isoformat()
            payload = {
                "version": "1.0",
                "model": "test",
                "generated_at": old_time,
                "payload_hash": "hash456",
                "events": [],
                "sources": {},
                "total_articles": 0,
            }
            repo.save(payload)
            cached = repo.get_cache_entry("hash456")
            self.assertIsNone(cached)


if __name__ == "__main__":
    unittest.main()
