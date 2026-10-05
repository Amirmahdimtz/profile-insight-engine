from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from fastapi import APIRouter, FastAPI

from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


@inject
class WebService:
    def __init__(self, config_reader: ConfigReader):
        self._api_prefix = config_reader.get_non_empty_string("api.prefix").rstrip("/")

    def create_app(self) -> FastAPI:
        app = FastAPI(title="Profile Insight Engine")
        application_root = Path(__file__).resolve().parent
        for feature_dir in sorted(application_root.iterdir(), key=lambda path: path.name):
            if not feature_dir.is_dir() or feature_dir.name.startswith("_"):
                continue
            controller_path = feature_dir / f"{feature_dir.name}_controller.py"
            if not controller_path.is_file():
                continue
            module = importlib.import_module(
                f"src.application.{feature_dir.name}.{feature_dir.name}_controller"
            )
            expected_name = "".join(part.capitalize() for part in feature_dir.name.split("_")) + "Controller"
            controller_type = getattr(module, expected_name, None)
            if not inspect.isclass(controller_type):
                raise RuntimeError(
                    f"controller class '{expected_name}' not found for feature '{feature_dir.name}'"
                )
            controller = controller_type()
            router = controller.api()
            if not isinstance(router, APIRouter):
                raise RuntimeError(f"controller '{expected_name}' api() must return APIRouter")
            app.include_router(
                router,
                prefix=f"{self._api_prefix}/{feature_dir.name}",
            )
        return app
