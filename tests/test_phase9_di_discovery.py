import json
import os
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class Phase9DiDiscoveryTests(unittest.TestCase):
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
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        routes = {tuple(item) for item in json.loads(completed.stdout)}
        self.assertIn(("POST", "/api/v1/profile_analysis/"), routes)
        self.assertIn(("GET", "/api/v1/profile_analysis/{analysis_id}"), routes)
        self.assertIn(("GET", "/api/v1/profile_analysis/{analysis_id}/result"), routes)


if __name__ == "__main__":
    unittest.main()
