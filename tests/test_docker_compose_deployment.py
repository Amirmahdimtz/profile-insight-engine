import pathlib
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT / "compose.yaml"
DOCKERFILE_PATH = ROOT / "Dockerfile"
NGINX_PATH = ROOT / "deploy/nginx/default.conf"
PREFLIGHT_PATH = ROOT / "deploy/preflight.sh"
DEPLOYMENT_DOC_PATH = ROOT / "docs/docker_compose_public_deployment.md"


class DockerComposeDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
        cls.dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
        cls.nginx = NGINX_PATH.read_text(encoding="utf-8")
        cls.preflight = PREFLIGHT_PATH.read_text(encoding="utf-8")
        cls.docs = DEPLOYMENT_DOC_PATH.read_text(encoding="utf-8")

    def test_only_reverse_proxy_publishes_public_ports(self):
        services = self.compose["services"]
        self.assertNotIn("ports", services["app"])
        self.assertNotIn("ports", services["db"])
        self.assertEqual(
            [port["published"] for port in services["proxy"]["ports"]],
            ["80", "443"],
        )
        self.assertTrue(self.compose["networks"]["backend"]["internal"])

    def test_app_waits_for_database_and_successful_migration(self):
        services = self.compose["services"]
        self.assertEqual(
            services["app"]["depends_on"]["db"]["condition"], "service_healthy"
        )
        self.assertEqual(
            services["app"]["depends_on"]["migrate"]["condition"],
            "service_completed_successfully",
        )
        self.assertIn("upgrade", services["migrate"]["command"])
        self.assertEqual(services["app"]["healthcheck"]["test"][0], "CMD")
        self.assertIn("APP_DB_USER", services["app"]["environment"]["PROFILE_INSIGHT_DATABASE_URL"])
        self.assertIn("POSTGRES_USER", services["migrate"]["environment"]["PROFILE_INSIGHT_DATABASE_URL"])
        self.assertIn("01-create-app-role.sh", str(services["db"]["volumes"]))
        self.assertIn("APP_DB_USER", services["db"]["environment"])
        self.assertIn("APP_DB_PASSWORD", services["db"]["environment"])

    def test_runtime_preserves_selected_ocr_profile_and_single_worker(self):
        self.assertIn("--workers", self.dockerfile)
        self.assertIn('"1"]', self.dockerfile)
        self.assertIn("PROFILE_INSIGHT_TESSDATA_DIR=/usr/local/share/tessdata_best", self.dockerfile)
        for language in ("fas", "eng", "osd"):
            self.assertIn(language, self.dockerfile)
        self.assertIn("USER 10001:10001", self.dockerfile)
        self.assertIn("TESSDATA_BEST_REVISION=e12c65a915945e4c28e237a9b52bc4a8f39a0cec", self.dockerfile)
        init_script = (ROOT / "deploy/postgres/init/01-create-app-role.sh").read_text(encoding="utf-8")
        self.assertIn('CREATE ROLE :"app_user" LOGIN PASSWORD :\'app_password\'', init_script)
        self.assertIn("ALTER DEFAULT PRIVILEGES FOR ROLE", init_script)

    def test_public_edge_requires_tls_and_basic_auth_and_limits_requests(self):
        for fragment in (
            "listen 443 ssl",
            "ssl_certificate /etc/nginx/tls/fullchain.pem",
            "ssl_certificate_key /etc/nginx/tls/privkey.pem",
            "auth_basic_user_file /etc/nginx/auth/.htpasswd",
            "limit_req zone=api_per_ip",
            "limit_conn api_conn_per_ip",
            "client_max_body_size 100m",
            "proxy_pass http://app:8000",
            'add_header Cache-Control "no-store" always',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.nginx)
        self.assertNotIn("$request_uri", self.nginx.split("log_format privacy_safe", 1)[1].split(";", 1)[0])

    def test_preflight_rejects_placeholder_secrets_and_validates_tls_and_compose(self):
        for fragment in (
            "POSTGRES_PASSWORD must be 64 lowercase hex characters",
            "APP_DB_PASSWORD must be 64 lowercase hex characters",
            "bootstrap/admin and application DB passwords must differ",
            "HTTP redirect hostname must match server_name",
            "openssl x509 -in deploy/tls/fullchain.pem -noout -checkhost",
            "TLS certificate and private key do not match",
            "must not be readable by group/other",
            "docker compose config --quiet",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.preflight)

    def test_documentation_calls_out_unverified_container_target_and_public_auth_boundary(self):
        for fragment in (
            "Linux container image and public-edge configuration are new deployment targets",
            "one application instance and one Uvicorn worker",
            "Basic Auth at Nginx is a shared access gate",
            "Certificate issuance and renewal are intentionally not coupled to Compose",
            "real selected-profile API/E2E benchmark",
            "no application-level authentication/authorization or CORS policy",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.docs)


if __name__ == "__main__":
    unittest.main()
