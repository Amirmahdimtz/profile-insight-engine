import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from src.infrastructure.di.bootstrap import _discover_module_names


ROOT = pathlib.Path(__file__).resolve().parents[1]
_FRESH_PROCESS_TIMEOUT_SECONDS = 60


class Phase9DiDiscoveryTests(unittest.TestCase):
    def test_discovery_scans_source_but_imports_only_inject_provider_modules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            (root / "provider.py").write_text(
                "from src.infrastructure.di.inject import inject\n"
                "@inject\n"
                "class Provider:\n"
                "    pass\n",
                encoding="utf-8",
            )
            (root / "heavy_module.py").write_text(
                "raise RuntimeError('must not be imported during DI discovery')\n",
                encoding="utf-8",
            )
            nested = root / "feature"
            nested.mkdir()
            (nested / "service.py").write_text(
                "from src.infrastructure.di.inject import inject\n"
                "@inject\n"
                "class Service:\n"
                "    pass\n",
                encoding="utf-8",
            )

            discovered = _discover_module_names("sample", (str(root),))

        self.assertEqual(
            discovered,
            ("sample.feature.service", "sample.provider"),
        )

    def test_discovery_resolves_web_stack_in_fresh_process_without_manual_imports(self):
        environment = dict(os.environ)
        environment["PROFILE_INSIGHT_DATABASE_URL"] = (
            "postgresql+asyncpg://postgres:phase9-test@127.0.0.1:1/profile_insight_phase9_test"
        )
        script = """
import json

from src.application.web import WebService
from src.infrastructure.di.bootstrap import bootstrap_di
from src.infrastructure.di.inject import resolve

bootstrap_di()
app = resolve(WebService).create_app()
document = app.openapi()
methods = {"get", "post", "put", "patch", "delete"}
routes = sorted(
    (method.upper(), path)
    for path, operations in document["paths"].items()
    for method in operations
    if method in methods
)
print(json.dumps(routes))
"""
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=_FRESH_PROCESS_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.fail(
                "fresh-process DI discovery exceeded the functional-test timeout; "
                f"stdout={exc.stdout!r}; stderr={exc.stderr!r}"
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        routes = {tuple(item) for item in json.loads(completed.stdout)}
        self.assertIn(("POST", "/api/v1/profile_analysis/"), routes)
        self.assertIn(("GET", "/api/v1/profile_analysis/{analysis_id}"), routes)
        self.assertIn(("GET", "/api/v1/profile_analysis/{analysis_id}/result"), routes)


if __name__ == "__main__":
    unittest.main()
