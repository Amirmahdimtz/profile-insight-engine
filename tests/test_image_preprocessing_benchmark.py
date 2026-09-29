import unittest
from unittest.mock import patch

from evaluation import image_preprocessing_benchmark as benchmark


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
