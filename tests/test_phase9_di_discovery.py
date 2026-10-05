import os
import unittest

from src.application.web import WebService
from src.infrastructure.di.bootstrap import bootstrap_di
from src.infrastructure.di.inject import resolve


_TEST_URL_ENV = "PROFILE_INSIGHT_PHASE9_TEST_DATABASE_URL"


@unittest.skipUnless(os.environ.get(_TEST_URL_ENV), f"set {_TEST_URL_ENV} to a disposable PostgreSQL database")
class Phase9DiDiscoveryTests(unittest.TestCase):
    def test_discovery_resolves_web_stack_without_manual_registration(self):
        previous = os.environ.get("PROFILE_INSIGHT_DATABASE_URL")
        os.environ["PROFILE_INSIGHT_DATABASE_URL"] = os.environ[_TEST_URL_ENV]
        try:
            bootstrap_di()
            app = resolve(WebService).create_app()
        finally:
            if previous is None:
                os.environ.pop("PROFILE_INSIGHT_DATABASE_URL", None)
            else:
                os.environ["PROFILE_INSIGHT_DATABASE_URL"] = previous
        routes = {
            (method, route.path)
            for route in app.routes
            for method in (route.methods or set())
            if route.path.startswith("/api/")
        }
        self.assertIn(("POST", "/api/v1/profile_analysis/"), routes)
        self.assertIn(("GET", "/api/v1/profile_analysis/{analysis_id}"), routes)
        self.assertIn(("GET", "/api/v1/profile_analysis/{analysis_id}/result"), routes)


if __name__ == "__main__":
    unittest.main()
