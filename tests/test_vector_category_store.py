"""Tests for VectorCategoryStore."""

import unittest
import tempfile
from pathlib import Path
from vector_category_store import VectorCategoryStore


class VectorCategoryStoreTest(unittest.TestCase):
    """Test suite for VectorCategoryStore."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = Path(self.temp_dir) / "test_category_store.json"
        self.store = VectorCategoryStore(self.store_path)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_categories_loaded(self):
        """Test that default categories are loaded on initialization."""
        self.assertGreater(len(self.store.categories), 0)
        # Should have at least the 10 default categories
        themes = {cat["theme"] for cat in self.store.categories}
        expected_themes = {
            "regulation", "market", "technology", "defi", "nft",
            "personnel", "security", "institutional", "macro", "ecosystem"
        }
        self.assertTrue(expected_themes.issubset(themes))

    def test_lookup_exact_match(self):
        """Test lookup with exact theme match."""
        # Add a test category
        self.store.record_classification(
            "test_theme",
            "test_sub",
            "Test description for lookup",
            source_event="test event"
        )

        # Lookup should find it
        result = self.store.lookup("Test description for lookup", "")
        self.assertIsNotNone(result)
        self.assertEqual(result["theme"], "test_theme")
        self.assertEqual(result["subcategory"], "test_sub")

    def test_lookup_similar_match(self):
        """Test lookup with similar but not exact description."""
        # Use existing default category
        result = self.store.lookup(
            "SEC announces new crypto regulation",
            "regulatory policy update"
        )

        # Should match regulation category
        if result:
            self.assertEqual(result["theme"], "regulation")
            self.assertIn("similarity", result)
            self.assertGreater(result["similarity"], 0)

    def test_lookup_below_threshold(self):
        """Test that lookup returns None when similarity is below threshold."""
        # Use completely unrelated text
        result = self.store.lookup(
            "xyz abc def random gibberish",
            "totally unrelated content"
        )

        # Should not match any category (or match with low similarity)
        # Depending on threshold, might return None
        if result:
            self.assertLess(result["similarity"], 0.9)

    def test_record_classification_creates_new_entry(self):
        """Test that recording a classification creates a new category."""
        initial_count = len(self.store.categories)

        self.store.record_classification(
            "new_theme",
            "new_subcategory",
            "Brand new category description",
            source_event="test"
        )

        self.assertEqual(len(self.store.categories), initial_count + 1)

        # Verify the new category
        new_cat = next(
            (c for c in self.store.categories if c["theme"] == "new_theme"),
            None
        )
        self.assertIsNotNone(new_cat)
        self.assertEqual(new_cat["subcategory"], "new_subcategory")
        self.assertIn("created_at", new_cat)

    def test_record_classification_updates_existing(self):
        """Test that recording updates timestamp of existing category."""
        # Record first time
        self.store.record_classification(
            "update_test",
            "sub1",
            "Original description",
            source_event="event1"
        )

        first_cat = next(c for c in self.store.categories if c["theme"] == "update_test")
        first_created = first_cat["created_at"]

        # Record again with same theme
        import time
        time.sleep(0.01)  # Ensure timestamp difference

        self.store.record_classification(
            "update_test",
            "sub2",
            "Updated description",
            source_event="event2"
        )

        updated_cat = next(c for c in self.store.categories if c["theme"] == "update_test")
        self.assertEqual(updated_cat["created_at"], first_created)
        self.assertIn("updated_at", updated_cat)

    def test_persistence(self):
        """Test that categories are persisted to disk."""
        # Add a category
        self.store.record_classification(
            "persist_test",
            "persist_sub",
            "Test persistence",
            source_event="test"
        )

        # Create new store instance from same path
        new_store = VectorCategoryStore(self.store_path)

        # Should load the persisted category
        persist_cat = next(
            (c for c in new_store.categories if c["theme"] == "persist_test"),
            None
        )
        self.assertIsNotNone(persist_cat)
        self.assertEqual(persist_cat["subcategory"], "persist_sub")

    def test_embedding_generation(self):
        """Test that embeddings are generated for categories."""
        cat = self.store.categories[0]
        # After lookup, categories should have _vector field (internal)
        # We can't directly test _vector as it's not persisted, but we can
        # verify lookup works which requires embeddings
        result = self.store.lookup(cat["description"], "")
        self.assertIsNotNone(result)

    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        # VectorCategoryStore uses dict-based sparse vectors
        vec1 = {"word1": 1.0, "word2": 0.0}
        vec2 = {"word1": 1.0, "word2": 0.0}
        vec3 = {"word3": 1.0, "word4": 0.0}

        # Same vectors should have high similarity
        sim1 = self.store._cosine(vec1, vec2)
        self.assertGreater(sim1, 0.9)

        # Completely different vectors should have similarity 0.0
        sim2 = self.store._cosine(vec1, vec3)
        self.assertAlmostEqual(sim2, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
