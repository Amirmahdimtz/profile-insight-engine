import math
import unittest

from src.core.profile_analysis.embedding_contracts import EmbeddingValidationError
from src.infrastructure.providers.embedding.transformers_embedding_provider import (
    TransformersEmbeddingProvider,
    _normalize_rows,
)


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


    def test_text_preprocessing_uses_siglip2_training_padding_semantics(self):
        calls = []

        class FakeInput:
            def to(self, device):
                self.device = device
                return self

        class FakeProcessor:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return {"input_ids": FakeInput()}

        class FakeFeatures:
            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [[1.0, 0.0], [0.0, 1.0]]

        class FakeModel:
            def get_text_features(self, **kwargs):
                return FakeFeatures()

        class FakeInferenceMode:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        class FakeTorch:
            @staticmethod
            def inference_mode():
                return FakeInferenceMode()

        provider = object.__new__(TransformersEmbeddingProvider)
        provider._batch_size = 8
        provider._processor = FakeProcessor()
        provider._device = "cpu"
        provider._torch = FakeTorch()
        provider._model = FakeModel()

        rows = provider._embed_texts_sync(("one", "two"))

        self.assertEqual(rows, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(calls[0]["text"], ["one", "two"])
        self.assertEqual(calls[0]["padding"], "max_length")
        self.assertTrue(calls[0]["truncation"])
        self.assertEqual(calls[0]["return_tensors"], "pt")


if __name__ == "__main__":
    unittest.main()
