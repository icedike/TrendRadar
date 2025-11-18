"""Tests for batch processing functionality."""

import unittest
from unittest.mock import Mock, patch
from ai_analyzer import AIAnalyzer


class BatchProcessingTest(unittest.TestCase):
    """Test suite for batch processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "ENABLED": True,
            "OLLAMA_MODEL": "llama3.2:3b",
            "OLLAMA_URL": "http://localhost:11434",
            "BATCH_SIZE": 5,  # Small batch size for testing
            "CACHE_TTL_HOURS": 24,
        }
        self.platform_configs = [
            {"id": "test_platform", "name": "Test Platform"}
        ]
        self.analyzer = AIAnalyzer(self.config, self.platform_configs)

    def test_small_dataset_no_batching(self):
        """Test that small datasets are processed without batching."""
        articles = [
            {"article_id": f"test:{i}:abc", "title": f"Article {i}", "platform_id": "test"}
            for i in range(3)  # Less than batch size
        ]

        with patch.object(self.analyzer.ollama_client, 'is_available', return_value=True):
            with patch.object(self.analyzer.ollama_client, 'generate') as mock_gen:
                mock_gen.return_value = '{"events": [{"event_id": "e1", "title": "Test", "article_refs": ["test:0:abc", "test:1:abc", "test:2:abc"], "rationale": "test"}]}'

                result = self.analyzer.cluster_events(articles)

                # Should call generate only once (no batching)
                self.assertEqual(mock_gen.call_count, 1)
                self.assertGreater(len(result), 0)

    def test_large_dataset_batching(self):
        """Test that large datasets are processed in batches."""
        # Create 12 articles (batch_size=5, so 3 batches needed)
        articles = [
            {"article_id": f"test:{i}:abc{i}", "title": f"Article {i}", "platform_id": "test"}
            for i in range(12)
        ]

        call_count = {"count": 0}

        def mock_generate(prompt, format=None):
            batch_num = call_count["count"]
            call_count["count"] += 1
            # Return different events for each batch
            return f'{{"events": [{{"event_id": "e{batch_num}", "title": "Batch {batch_num}", "article_refs": ["test:{batch_num}:abc{batch_num}"], "rationale": "test"}}]}}'

        with patch.object(self.analyzer, '_cluster_events_batched') as mock_batched:
            # Mock to return test events
            mock_batched.return_value = [
                {"event_id": "merged", "title": "Test", "articles": articles}
            ]

            with patch.object(self.analyzer.ollama_client, 'is_available', return_value=True):
                result = self.analyzer.cluster_events(articles)

                # Should use batched method
                mock_batched.assert_called_once()
                self.assertGreater(len(result), 0)

    def test_batched_clustering_merges_events(self):
        """Test that batched clustering merges identical normalized events across batches."""
        # Create articles with identical normalized titles (after removing symbols/spaces)
        articles = [
            {"article_id": "test:0:abc", "title": "Bitcoin Crashes Under 90K", "platform_id": "test"},
            {"article_id": "test:1:def", "title": "Bitcoin crashes under 90K!!!", "platform_id": "test"},
            {"article_id": "test:2:ghi", "title": "Other News Event", "platform_id": "test"},
            {"article_id": "test:3:jkl", "title": "Bitcoin Crashes UNDER 90K", "platform_id": "test"},
        ]

        # Split into 2 batches manually
        batch1 = articles[:2]
        batch2 = articles[2:]

        # Simulate separate clustering for each batch
        # These events have identical normalized titles: "bitcoincrashesunder90k"
        events_batch1 = [
            {"event_id": "e1", "title": "Bitcoin Crashes Under 90K", "articles": [batch1[0]]},
            {"event_id": "e2", "title": "Bitcoin crashes under 90K!!!", "articles": [batch1[1]]},
        ]
        events_batch2 = [
            {"event_id": "e3", "title": "Other News Event", "articles": [batch2[0]]},
            {"event_id": "e4", "title": "Bitcoin Crashes UNDER 90K", "articles": [batch2[1]]},
        ]

        all_events = events_batch1 + events_batch2

        # Test merging
        merged = self.analyzer._merge_cross_batch_events(all_events)

        # Should merge the 3 identical Bitcoin events into 1
        btc_events = [e for e in merged if "bitcoin" in e["title"].lower()]
        other_events = [e for e in merged if "other" in e["title"].lower()]

        self.assertEqual(len(btc_events), 1, "Should merge all identical Bitcoin events")
        self.assertEqual(len(other_events), 1)
        self.assertEqual(len(btc_events[0]["articles"]), 3, "Merged event should have 3 articles")

    def test_batch_processing_with_llm_failure(self):
        """Test that batch processing falls back to local clustering on failure."""
        articles = [
            {"article_id": f"test:{i}:abc{i}", "title": f"Article {i}", "platform_id": "test"}
            for i in range(8)  # 2 batches with batch_size=5
        ]

        with patch.object(self.analyzer.ollama_client, 'is_available', return_value=True):
            with patch.object(self.analyzer.ollama_client, 'generate') as mock_gen:
                # First batch fails 3 times, second batch succeeds
                mock_gen.side_effect = [
                    "Invalid JSON",  # Batch 1, attempt 1
                    "Still invalid",  # Batch 1, attempt 2
                    "Not JSON",  # Batch 1, attempt 3
                    '{"events": [{"event_id": "e1", "title": "Test", "article_refs": ["test:5:abc5"], "rationale": "test"}]}'  # Batch 2
                ]

                result = self.analyzer._cluster_events_batched(articles)

                # Should still return results (using local clustering for batch 1)
                self.assertGreater(len(result), 0)
                # Should have called generate 4 times (3 retries + 1 success)
                self.assertEqual(mock_gen.call_count, 4)

    def test_merge_cross_batch_empty_list(self):
        """Test that merging empty event list returns empty list."""
        result = self.analyzer._merge_cross_batch_events([])
        self.assertEqual(result, [])

    def test_merge_cross_batch_single_event(self):
        """Test that single event passes through unchanged."""
        events = [
            {"event_id": "e1", "title": "Test Event", "articles": [{"title": "Article"}]}
        ]
        result = self.analyzer._merge_cross_batch_events(events)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["event_id"], "e1")


if __name__ == "__main__":
    unittest.main()
