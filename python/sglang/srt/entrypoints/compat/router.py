"""FastAPI router for compatibility-layer adapters."""

from sglang.srt.entrypoints.compat.registry import build_compat_router

router, loaded_adapters = build_compat_router()
