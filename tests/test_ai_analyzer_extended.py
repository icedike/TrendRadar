"""Extended tests for AIAnalyzer - classification, scoring, and summarization."""

import unittest
from unittest.mock import Mock, patch
from ai_analyzer import AIAnalyzer


class AIAnalyzerExtendedTest(unittest.TestCase):
    """Extended test suite for AIAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "ENABLED": True,
            "OLLAMA_MODEL": "llama3.2:3b",
            "OLLAMA_URL": "http://localhost:11434",
            "BATCH_SIZE": 20,
            "CACHE_TTL_HOURS": 24,
        }
        self.platform_configs = [
            {"id": "test_platform", "name": "Test Platform"}
        ]
        self.analyzer = AIAnalyzer(self.config, self.platform_configs)

    def test_classify_theme_with_vector_store_hit(self):
        """Test classification when vector store finds a match."""
        event = {
            "title": "SEC announces new crypto regulation policy",
            "articles": []
        }

        # Mock vector store to return a hit
        with patch.object(self.analyzer.category_store, 'lookup') as mock_lookup:
            mock_lookup.return_value = {
                "theme": "regulation",
                "subcategory": "policy",
                "similarity": 0.95
            }

            result = self.analyzer.classify_theme(event)

            self.assertEqual(result["theme"], "regulation")
            self.assertEqual(result["subcategory"], "policy")
            self.assertEqual(result["classification_source"], "vector_store")
            self.assertIn("similarity", result)

    def test_classify_theme_with_llm(self):
        """Test classification using LLM when vector store misses."""
        event = {
            "title": "New blockchain technology breakthrough",
            "articles": []
        }

        # Mock vector store to return None (no match)
        # Mock ollama client to return valid JSON
        with patch.object(self.analyzer.category_store, 'lookup', return_value=None):
            with patch.object(self.analyzer.ollama_client, 'is_available', return_value=True):
                with patch.object(self.analyzer.ollama_client, 'generate') as mock_gen:
                    mock_gen.return_value = '{"theme": "technology", "subcategory": "innovation", "explanation": "Tech advancement"}'

                    result = self.analyzer.classify_theme(event)

                    self.assertEqual(result["theme"], "technology")
                    self.assertEqual(result["subcategory"], "innovation")
                    self.assertEqual(result["classification_source"], "llm")

    def test_classify_theme_keyword_fallback(self):
        """Test classification falls back to keywords when LLM fails."""
        event = {
            "title": "Bitcoin ETF trading volume surges",
            "articles": []
        }

        # Mock vector store miss and ollama unavailable
        with patch.object(self.analyzer.category_store, 'lookup', return_value=None):
            with patch.object(self.analyzer.ollama_client, 'is_available', return_value=False):
                result = self.analyzer.classify_theme(event)

                # Should match "market" due to "etf" and "trading" keywords
                self.assertEqual(result["theme"], "market")
                self.assertIn("classification_source", result)

    def test_score_importance_with_llm(self):
        """Test importance scoring with LLM."""
        event = {
            "articles": [
                {"title": "Major exchange hacked", "source_rank": 1},
                {"title": "Exchange hack confirmed", "source_rank": 2}
            ]
        }

        with patch.object(self.analyzer.ollama_client, 'is_available', return_value=True):
            with patch.object(self.analyzer.ollama_client, 'generate') as mock_gen:
                mock_gen.return_value = '{"importance": 9.5, "confidence": 0.95}'

                result = self.analyzer.score_importance(event)

                self.assertEqual(result["importance"], 9.5)
                self.assertEqual(result["confidence"], 0.95)

    def test_score_importance_heuristic_fallback(self):
        """Test importance scoring falls back to heuristic when LLM fails."""
        event = {
            "articles": [
                {"title": "News 1", "source_rank": 1},
                {"title": "News 2", "source_rank": 2},
                {"title": "News 3", "source_rank": 3}
            ]
        }

        with patch.object(self.analyzer.ollama_client, 'is_available', return_value=False):
            result = self.analyzer.score_importance(event)

            # Should use heuristic scoring
            self.assertIn("importance", result)
            self.assertIn("confidence", result)
            self.assertGreaterEqual(result["importance"], 1.0)
            self.assertLessEqual(result["importance"], 10.0)

    def test_score_importance_bounds(self):
        """Test that importance scores are properly bounded."""
        event = {
            "articles": [{"title": "Test", "source_rank": 1}]
        }

        with patch.object(self.analyzer.ollama_client, 'is_available', return_value=True):
            with patch.object(self.analyzer.ollama_client, 'generate') as mock_gen:
                # LLM returns out-of-bounds values
                mock_gen.return_value = '{"importance": 15.0, "confidence": 1.5}'

                result = self.analyzer.score_importance(event)

                # Should be clamped
                self.assertLessEqual(result["importance"], 10.0)
                self.assertLess(result["confidence"], 1.0)

    def test_analyze_sentiment_negative(self):
        """Test sentiment analysis for negative news."""
        event = {"title": "Major crypto exchange hacked and exploited"}

        result = self.analyzer.analyze_sentiment(event)

        self.assertEqual(result["sentiment"], "negative")

    def test_analyze_sentiment_positive(self):
        """Test sentiment analysis for positive news."""
        event = {"title": "Bitcoin reaches record high price surge"}

        result = self.analyzer.analyze_sentiment(event)

        self.assertEqual(result["sentiment"], "positive")

    def test_analyze_sentiment_neutral(self):
        """Test sentiment analysis for neutral news."""
        event = {"title": "New cryptocurrency wallet released"}

        result = self.analyzer.analyze_sentiment(event)

        self.assertEqual(result["sentiment"], "neutral")

    def test_generate_summary_with_llm(self):
        """Test summary generation with LLM."""
        event = {
            "articles": [
                {"title": "Bitcoin drops below $90K"},
                {"title": "BTC price crashes under 90000"}
            ]
        }

        with patch.object(self.analyzer.ollama_client, 'is_available', return_value=True):
            with patch.object(self.analyzer.ollama_client, 'generate') as mock_gen:
                mock_gen.return_value = "Bitcoin price dropped significantly below $90K amid market concerns."

                result = self.analyzer.generate_summary(event)

                self.assertGreater(len(result), 10)
                self.assertIn("Bitcoin", result)

    def test_generate_summary_fallback(self):
        """Test summary generation falls back to truncated context."""
        event = {
            "articles": [
                {"title": "Title 1"},
                {"title": "Title 2"},
                {"title": "Title 3"}
            ]
        }

        with patch.object(self.analyzer.ollama_client, 'is_available', return_value=False):
            result = self.analyzer.generate_summary(event)

            # Should be concatenated titles, truncated
            self.assertLessEqual(len(result), 240)
            self.assertIn("Title 1", result)

    def test_normalize_category(self):
        """Test category name normalization."""
        test_cases = [
            ("Market Analysis", "market_analysis"),
            ("DeFi & Lending", "defi_lending"),
            ("NFT-Collectibles", "nft_collectibles"),
            ("", "general"),
            ("  spaces  ", "spaces")
        ]

        for input_val, expected in test_cases:
            result = self.analyzer._normalize_category(input_val)
            self.assertEqual(result, expected)

    def test_normalize_text(self):
        """Test text normalization for clustering."""
        test_cases = [
            ("Bitcoin Crashes Under $90K", "bitcoincrashesunder90k"),
            ("Mt. Gox Moves $956M BTC", "mtgoxmoves956mbtc"),
            ("  Extra  Spaces  ", "extraspaces")
        ]

        for input_val, expected in test_cases:
            result = self.analyzer._normalize(input_val)
            self.assertEqual(result, expected)

    def test_hash_payload_consistency(self):
        """Test that identical payloads produce identical hashes."""
        payload1 = [
            {"article_id": "1", "title": "Test"},
            {"article_id": "2", "title": "News"}
        ]
        payload2 = [
            {"article_id": "1", "title": "Test"},
            {"article_id": "2", "title": "News"}
        ]

        hash1 = self.analyzer._hash_payload(payload1)
        hash2 = self.analyzer._hash_payload(payload2)

        self.assertEqual(hash1, hash2)

    def test_hash_payload_uniqueness(self):
        """Test that different payloads produce different hashes."""
        payload1 = [{"article_id": "1", "title": "Test"}]
        payload2 = [{"article_id": "2", "title": "Different"}]

        hash1 = self.analyzer._hash_payload(payload1)
        hash2 = self.analyzer._hash_payload(payload2)

        self.assertNotEqual(hash1, hash2)


if __name__ == "__main__":
    unittest.main()
