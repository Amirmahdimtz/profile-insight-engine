from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from evaluation.contracts import EvaluationDatasetManifest


def load_manifest(path: str | Path) -> EvaluationDatasetManifest:
    return EvaluationDatasetManifest.from_json(Path(path).read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a Phase 2 evaluation dataset manifest deterministically."
    )
    parser.add_argument("--manifest", required=True, help="Path to evaluation manifest JSON")
    parser.add_argument(
        "--dataset-root",
        help="Optional dataset root; when provided, referenced files must exist under it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.dataset_root:
        manifest.validate_references(args.dataset_root)
    print(
        f"VALID dataset_id={manifest.dataset_id} "
        f"dataset_version={manifest.dataset_version} "
        f"manifest_fingerprint={manifest.fingerprint()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
