from __future__ import annotations

import asyncio
import csv
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.profile_analysis.ocr_contracts import (
    OcrProviderBlock,
    OcrProviderError,
    OcrProviderResult,
    OcrRegion,
    OcrValidationError,
)
from src.core.profile_analysis.ocr_provider import IOcrProvider
from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


_REQUIRED_TSV_COLUMNS = {
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
}


@dataclass(frozen=True)
class TesseractOcrSettings:
    executable: str
    languages: tuple[str, ...]
    page_segmentation_mode: int
    timeout_seconds: int
    model_id: str
    config_version: str
    tessdata_dir: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.executable, str) or not self.executable or self.executable != self.executable.strip():
            raise OcrValidationError("ocr.executable must be a non-empty trimmed string")
        if not self.languages or any(
            not isinstance(item, str) or not item or item != item.strip() for item in self.languages
        ):
            raise OcrValidationError("ocr.languages must contain non-empty trimmed strings")
        if len(self.languages) != len(set(self.languages)):
            raise OcrValidationError("ocr.languages must not contain duplicates")
        if (
            isinstance(self.page_segmentation_mode, bool)
            or not isinstance(self.page_segmentation_mode, int)
            or self.page_segmentation_mode < 0
            or self.page_segmentation_mode > 13
        ):
            raise OcrValidationError("ocr.page_segmentation_mode must be an integer in [0, 13]")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, int) or self.timeout_seconds <= 0:
            raise OcrValidationError("ocr.timeout_seconds must be a positive integer")
        for value, name in ((self.model_id, "ocr.model_id"), (self.config_version, "ocr.config_version")):
            if not isinstance(value, str) or not value or value != value.strip():
                raise OcrValidationError(f"{name} must be a non-empty trimmed string")
        if self.tessdata_dir is not None:
            if not isinstance(self.tessdata_dir, str) or not self.tessdata_dir or self.tessdata_dir != self.tessdata_dir.strip():
                raise OcrValidationError("ocr.tessdata_dir must be null or a non-empty trimmed string")

    @classmethod
    def from_config(cls, config_reader: ConfigReader) -> "TesseractOcrSettings":
        provider = config_reader.get_non_empty_string("ocr.provider").lower()
        if provider != "tesseract":
            raise OcrValidationError("ocr.provider must be 'tesseract' for TesseractOcrProvider")
        return cls(
            executable=config_reader.get_non_empty_string("ocr.executable"),
            languages=config_reader.get_non_empty_string_list("ocr.languages"),
            page_segmentation_mode=config_reader.get_int_in_range(
                "ocr.page_segmentation_mode", minimum=0, maximum=13
            ),
            timeout_seconds=config_reader.get_positive_int("ocr.timeout_seconds"),
            model_id=config_reader.get_non_empty_string("ocr.model_id"),
            config_version=config_reader.get_non_empty_string("ocr.config_version"),
        )


@inject
class TesseractOcrProvider(IOcrProvider):
    def __init__(
        self,
        config_reader: ConfigReader,
        settings: TesseractOcrSettings | None = None,
    ):
        self._settings = settings or TesseractOcrSettings.from_config(config_reader)
        self._provider_version: str | None = None

    async def extract_async(self, image: CanonicalImage) -> OcrProviderResult:
        if not isinstance(image, CanonicalImage):
            raise OcrValidationError("image must be a CanonicalImage")
        path = Path(image.storage_reference)
        if not path.is_file():
            raise OcrProviderError("canonical image storage reference does not exist")
        try:
            tsv_text, provider_version = await asyncio.to_thread(self._run_tesseract, path)
            blocks = self._parse_tsv(tsv_text)
        except OcrProviderError:
            raise
        except (OSError, ValueError, csv.Error, UnicodeError) as exc:
            raise OcrProviderError("OCR provider returned malformed output") from exc
        return OcrProviderResult(
            image_id=image.image_id,
            blocks=blocks,
            provider="tesseract",
            provider_version=provider_version,
            model_id=self._settings.model_id,
            config_version=self._settings.config_version,
        )

    def _run_tesseract(self, image_path: Path) -> tuple[str, str]:
        command = [
            self._settings.executable,
            str(image_path),
            "stdout",
            "-l",
            "+".join(self._settings.languages),
            "--psm",
            str(self._settings.page_segmentation_mode),
        ]
        if self._settings.tessdata_dir is not None:
            command.extend(["--tessdata-dir", self._settings.tessdata_dir])
        command.append("tsv")
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._settings.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise OcrProviderError("Tesseract executable was not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise OcrProviderError("Tesseract OCR timed out") from exc
        if completed.returncode != 0:
            raise OcrProviderError("Tesseract OCR failed")
        try:
            output = completed.stdout.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise OcrProviderError("Tesseract OCR output is not valid UTF-8") from exc
        return output, self._get_provider_version()

    def _get_provider_version(self) -> str:
        if self._provider_version is not None:
            return self._provider_version
        try:
            completed = subprocess.run(
                [self._settings.executable, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._settings.timeout_seconds,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise OcrProviderError("unable to determine Tesseract version") from exc
        if completed.returncode != 0:
            raise OcrProviderError("unable to determine Tesseract version")
        try:
            first_line = completed.stdout.decode("utf-8", errors="strict").splitlines()[0].strip()
        except (UnicodeDecodeError, IndexError) as exc:
            raise OcrProviderError("unable to determine Tesseract version") from exc
        if not first_line:
            raise OcrProviderError("unable to determine Tesseract version")
        self._provider_version = first_line
        return first_line

    @staticmethod
    def _parse_tsv(tsv_text: str) -> tuple[OcrProviderBlock, ...]:
        reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
        if reader.fieldnames is None or not _REQUIRED_TSV_COLUMNS.issubset(set(reader.fieldnames)):
            raise OcrProviderError("Tesseract TSV output is missing required columns")

        grouped: dict[tuple[int, int, int, int], list[tuple[str, float, OcrRegion]]] = {}
        order_keys: list[tuple[int, int, int, int]] = []
        for row in reader:
            try:
                level = int(row["level"])
            except (TypeError, ValueError) as exc:
                raise OcrProviderError("Tesseract TSV level is invalid") from exc
            if level != 5:
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            try:
                confidence_raw = float(row["conf"])
                if confidence_raw < 0.0 or confidence_raw > 100.0:
                    raise ValueError
                region = OcrRegion(
                    x=int(row["left"]),
                    y=int(row["top"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
                key = (
                    int(row["page_num"]),
                    int(row["block_num"]),
                    int(row["par_num"]),
                    int(row["line_num"]),
                )
            except (TypeError, ValueError, OcrValidationError) as exc:
                raise OcrProviderError("Tesseract TSV word row is invalid") from exc
            if key not in grouped:
                grouped[key] = []
                order_keys.append(key)
            grouped[key].append((text, confidence_raw / 100.0, region))

        blocks: list[OcrProviderBlock] = []
        for order, key in enumerate(order_keys):
            words = grouped[key]
            min_x = min(item[2].x for item in words)
            min_y = min(item[2].y for item in words)
            max_x = max(item[2].x + item[2].width for item in words)
            max_y = max(item[2].y + item[2].height for item in words)
            blocks.append(
                OcrProviderBlock(
                    order=order,
                    raw_text=" ".join(item[0] for item in words),
                    confidence=sum(item[1] for item in words) / len(words),
                    region=OcrRegion(
                        x=min_x,
                        y=min_y,
                        width=max_x - min_x,
                        height=max_y - min_y,
                    ),
                )
            )
        return tuple(blocks)
