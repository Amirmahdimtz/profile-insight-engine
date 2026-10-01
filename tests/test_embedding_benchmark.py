import unittest
from types import SimpleNamespace

from evaluation.embedding_benchmark import _calculate_semantic_metrics, _percentile
from src.core.profile_analysis.embedding_contracts import (
    SemanticRetrievalResult,
    SemanticThemeResult,
    ThemeSimilarityEvidence,
)


def sample(image_id, labels, slices):
    return SimpleNamespace(
        image=SimpleNamespace(image_id=image_id),
        ground_truth=SimpleNamespace(
            labels=tuple(SimpleNamespace(label=label) for label in labels)
        ),
        slices=tuple(SimpleNamespace(value=value) for value in slices),
    )


class EmbeddingBenchmarkTests(unittest.TestCase):
    def test_semantic_metrics_cover_retrieval_theme_f1_and_language_slices(self):
        samples = (
            sample("a", ("football",), ("english_heavy",)),
            sample("b", ("فوتبال",), ("persian_heavy",)),
        )
        result = SemanticThemeResult(
            theme_similarities=(
                ThemeSimilarityEvidence("a", "football", 0.9, 0.5),
                ThemeSimilarityEvidence("a", "فوتبال", 0.1, 0.5),
                ThemeSimilarityEvidence("b", "football", 0.2, 0.5),
                ThemeSimilarityEvidence("b", "فوتبال", 0.8, 0.5),
            ),
            retrieval_results=(
                SemanticRetrievalResult("football", ("a", "b"), (0.9, 0.2)),
                SemanticRetrievalResult("فوتبال", ("b", "a"), (0.8, 0.1)),
            ),
            near_duplicate_groups=(),
            provider="test",
            provider_version="1",
            model_id="model",
            model_version="rev",
            config_version="cfg",
            embedding_dimension=2,
        )
        metrics = _calculate_semantic_metrics(
            samples, result, ("football", "فوتبال"), recall_k=1
        )
        self.assertEqual(metrics["theme_macro_f1"], 1.0)
        self.assertEqual(metrics["mean_recall_at_k"], 1.0)
        self.assertEqual(metrics["map"], 1.0)
        self.assertEqual(
            metrics["cross_language_theme_macro_f1"]["english_heavy"], 1.0
        )
        self.assertEqual(
            metrics["cross_language_theme_macro_f1"]["persian_heavy"], 1.0
        )
        self.assertIsNone(
            metrics["cross_language_theme_macro_f1"]["mixed_persian_english"]
        )

    def test_percentile_is_deterministic(self):
        self.assertEqual(_percentile([30.0, 10.0, 20.0], 0.5), 20.0)
        self.assertEqual(_percentile([], 0.95), 0.0)


if __name__ == "__main__":
    unittest.main()
