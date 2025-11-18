import tempfile
import unittest

from ai_analyzer import AIAnalyzer, AIResultRepository


class AIAnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.platforms = [
            {"id": "source_a", "name": "Source A"},
            {"id": "source_b", "name": "Source B"},
        ]
        self.config = {
            "ENABLED": False,
            "OLLAMA_MODEL": "test",
            "OLLAMA_URL": "http://localhost:11434",
            "BATCH_SIZE": 10,
            "CACHE_TTL_HOURS": 24,
            "OUTPUT_VERSION": "1.0",
            "OUTPUT_ROOT": self.temp_dir.name,
        }
        self.repository = AIResultRepository(output_root=self.temp_dir.name)
        self.analyzer = AIAnalyzer(self.config, self.platforms, self.repository)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prepare_ai_payload(self) -> None:
        raw_results = {
            "source_a": {
                "Breaking News": {"ranks": [1], "url": "https://a"}
            },
            "source_b": {
                "Breaking News": {"ranks": [2], "url": "https://b"}
            },
        }
        title_info = {
            "source_a": {"Breaking News": {"last_time": "2025-01-01 00:00"}},
            "source_b": {"Breaking News": {"last_time": "2025-01-01 00:00"}},
        }
        payload = self.analyzer.prepare_ai_payload(raw_results, title_info)
        self.assertEqual(len(payload), 2)
        self.assertTrue(all("article_id" in article for article in payload))

    def test_cluster_events_without_ollama(self) -> None:
        articles = [
            {
                "article_id": "source_a:1",
                "platform_id": "source_a",
                "platform_name": "Source A",
                "title": "Bitcoin surges",
                "ranks": [1],
            },
            {
                "article_id": "source_b:1",
                "platform_id": "source_b",
                "platform_name": "Source B",
                "title": "Bitcoin surges!",
                "ranks": [2],
            },
        ]
        events = self.analyzer.cluster_events(articles)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(events[0]["articles"]), 2)


if __name__ == "__main__":
    unittest.main()
