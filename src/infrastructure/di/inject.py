from __future__ import annotations

import inspect
import threading
from functools import wraps
from typing import Any, TypeVar, get_type_hints


T = TypeVar("T")


class DependencyResolutionError(RuntimeError):
    """Raised when constructor injection cannot resolve a required dependency."""


_providers: dict[type[Any], type[Any]] = {}
_singletons: dict[type[Any], Any] = {}
_lock = threading.RLock()


def _register_provider(provider_type: type[Any]) -> None:
    with _lock:
        _providers[provider_type] = provider_type


def inject(cls: type[T]) -> type[T]:
    """Mark a class as a DI provider and inject omitted required constructor dependencies."""

    original_init = cls.__init__
    signature = inspect.signature(original_init)

    @wraps(original_init)
    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind_partial(self, *args, **kwargs)
        try:
            type_hints = get_type_hints(original_init)
        except (NameError, TypeError) as exc:
            raise DependencyResolutionError(
                f"cannot resolve type hints for {cls.__module__}.{cls.__qualname__}"
            ) from exc

        injected_kwargs = dict(kwargs)
        for name, parameter in signature.parameters.items():
            if name == "self" or name in bound.arguments:
                continue
            if parameter.default is not inspect.Parameter.empty:
                continue
            annotation = type_hints.get(name)
            if annotation is None:
                raise DependencyResolutionError(
                    f"required dependency '{name}' on {cls.__qualname__} needs a type annotation"
                )
            injected_kwargs[name] = resolve(annotation)
        original_init(self, *args, **injected_kwargs)

    cls.__init__ = wrapped_init  # type: ignore[method-assign]
    setattr(cls, "__di_provider__", True)
    _register_provider(cls)
    return cls


def resolve(dependency_type: type[T]) -> T:
    """Resolve a discovered provider by concrete type."""

    with _lock:
        provider_type = _providers.get(dependency_type)
        if provider_type is None:
            raise DependencyResolutionError(
                f"no provider registered for {dependency_type!r}; call bootstrap_di() first"
            )
        if getattr(provider_type, "__di_singleton__", False):
            instance = _singletons.get(provider_type)
            if instance is None:
                instance = provider_type()
                _singletons[provider_type] = instance
            return instance
    return provider_type()
