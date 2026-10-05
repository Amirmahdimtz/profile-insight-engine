from src.application.web import WebService
from src.infrastructure.di.bootstrap import bootstrap_di
from src.infrastructure.di.inject import resolve


bootstrap_di()
app = resolve(WebService).create_app()
