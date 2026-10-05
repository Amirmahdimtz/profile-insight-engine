import tempfile
import unittest
from pathlib import Path

from evaluation.insight_benchmark import (
    METRIC_SCOPE,
    REPORT_SCHEMA_VERSION,
    run_benchmark,
)
from src.infrastructure.utils.config_reader import ConfigReader


CONFIG = """\
insights:
  policy_version: phase8-test-v1
  minimum_evidence_count: 2
  minimum_unique_image_count: 2
  minimum_image_coverage: 0.1
  minimum_source_diversity: 1
  minimum_cross_image_consistency: 0.0
  minimum_average_confidence: 0.0
  source_diversity_saturation: 2
"""


class InsightBenchmarkTests(unittest.TestCase):
    def _config(self):
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False)
        handle.write(CONFIG)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return ConfigReader(handle.name)

    def test_contract_benchmark_is_reproducible_and_marks_quality_limitations(self):
        first = run_benchmark(self._config())
        second = run_benchmark(self._config())
        self.assertEqual(first.schema_version, REPORT_SCHEMA_VERSION)
        self.assertEqual(first.metric_scope, METRIC_SCOPE)
        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.insight_precision, 1.0)
        self.assertEqual(first.insight_recall, 1.0)
        self.assertEqual(first.false_unsupported_insight_rate, 0.0)
        self.assertEqual(first.summary_factuality, 1.0)
        self.assertIsNone(first.confidence_calibration)
        self.assertIsNone(first.human_review_agreement)
        self.assertTrue(first.deterministic_rerun)
        self.assertTrue(first.reversed_input_deterministic)
        self.assertTrue(first.limitations)


if __name__ == "__main__":
    unittest.main()
