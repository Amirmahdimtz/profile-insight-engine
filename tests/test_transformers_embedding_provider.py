import math
import unittest

from src.core.profile_analysis.embedding_contracts import EmbeddingValidationError
from src.infrastructure.providers.embedding.transformers_embedding_provider import _normalize_rows


class TransformersEmbeddingProviderTests(unittest.TestCase):
    def test_normalize_rows_preserves_order_and_normalizes(self):
        vectors = _normalize_rows(((3.0, 4.0), (0.0, 2.0)), ("a", "b"))
        self.assertEqual(tuple(item.item_id for item in vectors), ("a", "b"))
        self.assertAlmostEqual(vectors[0].values[0], 0.6)
        self.assertAlmostEqual(vectors[0].values[1], 0.8)
        self.assertEqual(vectors[1].values, (0.0, 1.0))

    def test_invalid_runtime_vectors_are_rejected(self):
        for rows in (((0.0, 0.0),), ((math.nan, 1.0),), ((math.inf, 1.0),), ((),)):
            with self.subTest(rows=rows):
                with self.assertRaises(EmbeddingValidationError):
                    _normalize_rows(rows, ("a",))

    def test_runtime_output_count_must_match_input_count(self):
        with self.assertRaises(EmbeddingValidationError):
            _normalize_rows(((1.0, 0.0),), ("a", "b"))


if __name__ == "__main__":
    unittest.main()
