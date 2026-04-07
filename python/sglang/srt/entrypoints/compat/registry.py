from __future__ import annotations

import importlib
import os
from typing import Callable, Iterable, List, Mapping, Sequence

from fastapi import APIRouter, Request

from sglang.srt.entrypoints.compat.base import (
    CompatAdapter,
    CompatRouteSpec,
    CompatServiceRegistry,
)
from sglang.srt.entrypoints.compat.bailian_qwen3_rerank import (
    BailianQwen3RerankAdapter,
)

_BUILTIN_COMPAT_ADAPTERS: dict[str, Callable[[], CompatAdapter]] = {
    "bailian_qwen3_rerank": BailianQwen3RerankAdapter,
}


def _merge_unique_adapters(
    adapters: List[CompatAdapter],
    new_adapters: Iterable[CompatAdapter],
) -> None:
    existing_names = {adapter.name for adapter in adapters}
    for adapter in new_adapters:
        if adapter.name in existing_names:
            continue
        adapters.append(adapter)
        existing_names.add(adapter.name)


def _normalize_loaded_adapters(obj) -> List[CompatAdapter]:
    if isinstance(obj, CompatAdapter):
        return [obj]
    if callable(obj):
        return _normalize_loaded_adapters(obj())
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        adapters = list(obj)
        if not all(isinstance(adapter, CompatAdapter) for adapter in adapters):
            raise TypeError(
                "Every loaded compatibility adapter must inherit CompatAdapter."
            )
        return adapters
    raise TypeError("Unsupported compatibility adapter entry.")


def _load_adapter_entry(entry: str) -> List[CompatAdapter]:
    if entry in _BUILTIN_COMPAT_ADAPTERS:
        return [_BUILTIN_COMPAT_ADAPTERS[entry]()]

    module_name, _, attr_name = entry.partition(":")
    if not module_name or not attr_name:
        raise ValueError(
            "Compatibility adapter entries must be a built-in adapter name or "
            "use the format 'module.path:object'."
        )
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    return _normalize_loaded_adapters(obj)


def load_compat_adapters(
    extra_entries: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> List[CompatAdapter]:
    """Load built-in adapters plus optional custom adapters.

    Users can provide custom adapters through `SGLANG_COMPAT_ADAPTERS`, a
    comma-separated list of built-in adapter names or `module.path:object`
    entries.
    """
    env_map = os.environ if env is None else env
    adapters: List[CompatAdapter] = [
        factory() for factory in _BUILTIN_COMPAT_ADAPTERS.values()
    ]

    env_entries = [
        entry.strip()
        for entry in env_map.get("SGLANG_COMPAT_ADAPTERS", "").split(",")
        if entry.strip()
    ]
    for entry in [*(extra_entries or []), *env_entries]:
        _merge_unique_adapters(adapters, _load_adapter_entry(entry))

    return adapters


def _validate_unique_routes(adapters: Sequence[CompatAdapter]) -> None:
    seen: dict[tuple[str, tuple[str, ...]], str] = {}
    for adapter in adapters:
        for route in adapter.route_specs():
            key = (route.path, tuple(route.methods))
            if key in seen:
                raise ValueError(
                    f"Duplicate compatibility route {route.path} {route.methods} "
                    f"registered by '{adapter.name}' and '{seen[key]}'."
                )
            seen[key] = adapter.name


def build_compat_router(
    adapters: Sequence[CompatAdapter] | None = None,
) -> tuple[APIRouter, List[CompatAdapter]]:
    loaded_adapters = list(adapters) if adapters is not None else load_compat_adapters()
    _validate_unique_routes(loaded_adapters)

    router = APIRouter()

    for adapter in loaded_adapters:
        for route in adapter.route_specs():
            router.add_api_route(
                route.path,
                _make_handler(adapter),
                methods=list(route.methods),
                include_in_schema=route.include_in_schema,
                name=f"compat_{adapter.name}_{_route_name_suffix(route)}",
            )

    return router, loaded_adapters


def _route_name_suffix(route: CompatRouteSpec) -> str:
    return route.path.strip("/").replace("/", "_") or "root"


def _make_handler(adapter: CompatAdapter):
    async def handler(raw_request: Request):
        services = getattr(
            raw_request.app.state,
            "compat_service_registry",
            CompatServiceRegistry(),
        )
        return await adapter.handle_request(raw_request, services)

    return handler
