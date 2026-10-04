import math
import unittest

from src.core.profile_analysis.embedding_contracts import (
    EmbeddingBatchResult,
    EmbeddingValidationError,
    EmbeddingVector,
)


class EmbeddingContractTests(unittest.TestCase):
    def test_valid_vector_is_normalized_and_dimension_is_exposed(self):
        vector = EmbeddingVector(item_id="img-1", values=(0.6, 0.8))
        self.assertEqual(vector.dimension, 2)

    def test_nan_inf_zero_norm_and_non_normalized_are_rejected(self):
        for values in ((math.nan, 1.0), (math.inf, 1.0), (0.0, 0.0), (1.0, 1.0)):
            with self.subTest(values=values):
                with self.assertRaises(EmbeddingValidationError):
                    EmbeddingVector(item_id="x", values=values)

    def test_batch_requires_consistent_dimension_and_unique_ids(self):
        with self.assertRaises(EmbeddingValidationError):
            EmbeddingBatchResult(
                vectors=(
                    EmbeddingVector("a", (1.0, 0.0)),
                    EmbeddingVector("b", (1.0, 0.0, 0.0)),
                ),
                provider="p",
                provider_version="1",
                model_id="m",
                model_version="v",
                config_version="c",
            )
        with self.assertRaises(EmbeddingValidationError):
            EmbeddingBatchResult(
                vectors=(
                    EmbeddingVector("a", (1.0, 0.0)),
                    EmbeddingVector("a", (0.0, 1.0)),
                ),
                provider="p",
                provider_version="1",
                model_id="m",
                model_version="v",
                config_version="c",
            )

    def test_semantic_result_rejects_invalid_collection_items(self):
        from src.core.profile_analysis.embedding_contracts import SemanticThemeResult

        with self.assertRaises(EmbeddingValidationError):
            SemanticThemeResult(
                theme_similarities=(object(),),
                retrieval_results=(),
                near_duplicate_groups=(),
                provider="provider",
                provider_version="1.0",
                model_id="model",
                model_version="revision",
                config_version="config",
                embedding_dimension=2,
            )


if __name__ == "__main__":
    unittest.main()
