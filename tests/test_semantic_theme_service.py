import asyncio
import tempfile
import unittest
from pathlib import Path

from src.core.profile_analysis.embedding_contracts import EmbeddingBatchResult, EmbeddingVector, EmbeddingValidationError
from src.core.profile_analysis.embedding_provider import IEmbeddingProvider
from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.services.profile_analysis.semantic_theme_service import SemanticThemeService
from src.infrastructure.utils.config_reader import ConfigReader


class FakeEmbeddingProvider(IEmbeddingProvider):
    def __init__(self, image_vectors, text_vectors):
        self.image_vectors = image_vectors
        self.text_vectors = text_vectors

    async def embed_images_async(self, images):
        return self._batch(tuple(EmbeddingVector(i.image_id, self.image_vectors[i.image_id]) for i in images))

    async def embed_texts_async(self, texts):
        return self._batch(tuple(EmbeddingVector(text, self.text_vectors[text]) for text in texts))

    @staticmethod
    def _batch(vectors):
        return EmbeddingBatchResult(
            vectors=vectors,
            provider="fake-test-provider",
            provider_version="1",
            model_id="test-model",
            model_version="test-revision",
            config_version="phase6-test",
        )


def image(image_id):
    return CanonicalImage(
        image_id=image_id,
        content_hash=("a" if image_id == "a" else "b") * 64,
        storage_reference=f"/tmp/{image_id}.png",
        mime_type="image/png",
        width=10,
        height=10,
        size_bytes=10,
        original_mime_type="image/png",
        original_width=10,
        original_height=10,
        original_size_bytes=10,
        was_resized=False,
        exif_orientation_applied=False,
    )


class SemanticThemeServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        config_path = Path(self.temp.name) / "config.yaml"
        config_path.write_text(
            "embedding:\n"
            "  theme_similarity_threshold: 0.8\n"
            "  near_duplicate_similarity_threshold: 0.99\n"
            "  retrieval_k: 2\n",
            encoding="utf-8",
        )
        self.config = ConfigReader(config_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_threshold_boundaries_retrieval_and_duplicate_groups_are_deterministic(self):
        provider = FakeEmbeddingProvider(
            image_vectors={"a": (1.0, 0.0), "b": (0.99995, 0.01)},
            text_vectors={"football": (1.0, 0.0)},
        )
        service = SemanticThemeService(provider, self.config)
        result = asyncio.run(service.analyze_async((image("a"), image("b")), ("football",)))
        self.assertEqual([x.image_id for x in result.theme_similarities], ["a", "b"])
        self.assertTrue(all(x.matched for x in result.theme_similarities))
        self.assertEqual(result.retrieval_results[0].ranked_image_ids, ("a", "b"))
        self.assertEqual(result.near_duplicate_groups[0].image_ids, ("a", "b"))

    def test_provider_order_mismatch_is_rejected(self):
        class ReversingProvider(FakeEmbeddingProvider):
            async def embed_images_async(self, images):
                items = list(images)
                items.reverse()
                return self._batch(tuple(EmbeddingVector(i.image_id, self.image_vectors[i.image_id]) for i in items))

        provider = ReversingProvider(
            image_vectors={"a": (1.0, 0.0), "b": (0.0, 1.0)},
            text_vectors={"football": (1.0, 0.0)},
        )
        service = SemanticThemeService(provider, self.config)
        with self.assertRaises(EmbeddingValidationError):
            asyncio.run(service.analyze_async((image("a"), image("b")), ("football",)))

    def test_theme_threshold_is_inclusive_at_exact_boundary(self):
        provider = FakeEmbeddingProvider(
            image_vectors={"a": (0.8, 0.6)},
            text_vectors={"football": (1.0, 0.0)},
        )
        service = SemanticThemeService(provider, self.config)
        result = asyncio.run(service.analyze_async((image("a"),), ("football",)))
        self.assertAlmostEqual(result.theme_similarities[0].similarity, 0.8)
        self.assertTrue(result.theme_similarities[0].matched)

    def test_concurrent_calls_are_deterministic(self):
        provider = FakeEmbeddingProvider(
            image_vectors={"a": (1.0, 0.0), "b": (0.0, 1.0)},
            text_vectors={"football": (1.0, 0.0)},
        )
        service = SemanticThemeService(provider, self.config)

        async def run_both():
            return await asyncio.gather(
                service.analyze_async((image("a"), image("b")), ("football",)),
                service.analyze_async((image("a"), image("b")), ("football",)),
            )

        first, second = asyncio.run(run_both())
        self.assertEqual(first, second)

    def test_sensitive_person_level_theme_is_rejected(self):
        provider = FakeEmbeddingProvider(
            image_vectors={"a": (1.0, 0.0)},
            text_vectors={"person religion": (1.0, 0.0)},
        )
        service = SemanticThemeService(provider, self.config)
        with self.assertRaises(EmbeddingValidationError):
            asyncio.run(service.analyze_async((image("a"),), ("person religion",)))

    def test_duplicate_content_with_distinct_ids_preserves_both_inputs(self):
        provider = FakeEmbeddingProvider(
            image_vectors={"a": (1.0, 0.0), "b": (1.0, 0.0)},
            text_vectors={"travel": (0.0, 1.0)},
        )
        service = SemanticThemeService(provider, self.config)
        result = asyncio.run(service.analyze_async((image("a"), image("b")), ("travel",)))
        self.assertEqual(result.near_duplicate_groups[0].image_ids, ("a", "b"))


if __name__ == "__main__":
    unittest.main()
