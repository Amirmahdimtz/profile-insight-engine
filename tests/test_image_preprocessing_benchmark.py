import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import image_preprocessing_benchmark as benchmark


class ImagePreprocessingBenchmarkWorkloadTests(unittest.TestCase):
    def test_small_dataset_is_replayed_deterministically_for_larger_workload(self):
        indices = benchmark._workload_sample_indices(17, 50)

        self.assertEqual(len(indices), 50)
        self.assertEqual(indices[:17], tuple(range(17)))
        self.assertEqual(indices[17:34], tuple(range(17)))
        self.assertEqual(indices[34:], tuple(range(16)))
        self.assertEqual(len({_id for _id in (benchmark._benchmark_image_id(i) for i in range(50))}), 50)

    def test_workload_selection_rejects_invalid_sizes(self):
        with self.assertRaisesRegex(ValueError, "dataset_size"):
            benchmark._workload_sample_indices(0, 1)
        with self.assertRaisesRegex(ValueError, "image_count"):
            benchmark._workload_sample_indices(1, 0)


class ImagePreprocessingBenchmarkSmallDatasetTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_benchmark_replays_authorized_samples_for_20_and_50_image_batches(self):
        calls = []

        class FakeService:
            async def process_async(self, request, inputs):
                calls.append((request, inputs))
                return SimpleNamespace(images=(), duplicates=())

            async def release_async(self, batch):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            samples = []
            for index in range(17):
                relative_path = f"sample-{index}.png"
                (dataset_root / relative_path).write_bytes(b"authorized-image-bytes")
                samples.append(
                    SimpleNamespace(
                        image=SimpleNamespace(
                            image_id=f"source-{index}",
                            relative_path=relative_path,
                        )
                    )
                )

            manifest = SimpleNamespace(
                dataset_id="phase3-small-dataset",
                dataset_version="1.0.0",
                samples=tuple(samples),
                validate_references=lambda root: "content-fingerprint",
                fingerprint=lambda: "manifest-fingerprint",
            )
            fake_service = FakeService()
            fake_config = SimpleNamespace(get_positive_int=lambda key: 2048)

            with (
                patch.object(benchmark, "ConfigReader", return_value=fake_config),
                patch.object(benchmark, "LocalImageStorage", return_value=object()),
                patch.object(benchmark, "ImageProcessingService", return_value=fake_service),
                patch.object(benchmark, "_declared_mime", return_value="image/png"),
                patch.object(benchmark, "_measure_decode_throughput", return_value=100.0),
                patch.object(benchmark, "_measure_resize_cost", return_value=(0.0, 0)),
                patch.object(benchmark, "_peak_rss_mb", return_value=128.0),
            ):
                report = await benchmark.run_benchmark_async(
                    manifest,
                    dataset_root,
                    (20, 50),
                    1,
                )

        self.assertEqual(report.schema_version, "1.1.0")
        self.assertEqual(report.dataset_sample_count, 17)
        self.assertEqual(
            [(item.image_count, item.source_sample_count, item.replayed_sample_count) for item in report.batch_results],
            [(20, 17, 3), (50, 17, 33)],
        )
        self.assertEqual([len(inputs) for _, inputs in calls], [20, 50])
        for request, inputs in calls:
            self.assertEqual(len(request.image_ids), len(set(request.image_ids)))
            self.assertEqual(
                request.image_ids,
                tuple(item.image_id for item in inputs),
            )


class ImagePreprocessingBenchmarkPlatformTests(unittest.TestCase):
    def test_peak_rss_uses_windows_working_set_measurement_on_windows(self):
        with (
            patch.object(benchmark.sys, "platform", "win32"),
            patch.object(
                benchmark,
                "_windows_peak_working_set_mb",
                return_value=321.5,
            ) as windows_measurement,
        ):
            self.assertEqual(benchmark._peak_rss_mb(), 321.5)

        windows_measurement.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
