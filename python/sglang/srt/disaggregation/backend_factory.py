# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Type

from sglang.srt.disaggregation.utils import KVClassType
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class DisaggBackendProvider(Protocol):
    """Dynamic backend provider protocol for PD disaggregation transfer."""

    def get_kv_class(self, class_type: KVClassType) -> Optional[Type]: ...


@dataclass(frozen=True)
class DynamicBackendConfig:
    backend_name: str
    module_path: str
    class_name: str

    @classmethod
    def from_json_str(cls, s: Optional[str]) -> "DynamicBackendConfig":
        if not s:
            raise ValueError("Dynamic backend requires a non-empty JSON extra config.")
        try:
            obj = json.loads(s)
        except Exception as e:
            raise ValueError(
                f"Invalid JSON for dynamic backend extra config: {e}"
            ) from e
        if not isinstance(obj, dict):
            raise ValueError("Dynamic backend extra config must be a JSON object.")
        for k in ("backend_name", "module_path", "class_name"):
            if k not in obj:
                raise ValueError(
                    f"Missing required field '{k}' in dynamic backend config."
                )
        return cls(
            backend_name=str(obj["backend_name"]),
            module_path=str(obj["module_path"]),
            class_name=str(obj["class_name"]),
        )


class DisaggBackendFactory:
    """Factory for creating dynamic disaggregation backend providers."""

    _provider_cache: Dict[str, DisaggBackendProvider] = {}

    @classmethod
    def get_provider(cls, server_args: ServerArgs) -> DisaggBackendProvider:
        cfg = DynamicBackendConfig.from_json_str(
            getattr(server_args, "disaggregation_transfer_backend_extra_config", None)
        )
        cache_key = f"{cfg.module_path}:{cfg.class_name}"
        if cache_key in cls._provider_cache:
            return cls._provider_cache[cache_key]

        logger.info(
            "Loading dynamic disaggregation backend '%s' (%s.%s)",
            cfg.backend_name,
            cfg.module_path,
            cfg.class_name,
        )
        module = importlib.import_module(cfg.module_path)
        provider_cls = getattr(module, cfg.class_name)
        provider = provider_cls()
        if not hasattr(provider, "get_kv_class"):
            raise TypeError(
                f"Dynamic backend provider {cfg.module_path}.{cfg.class_name} "
                f"must implement get_kv_class(class_type: KVClassType) -> Optional[type]."
            )
        cls._provider_cache[cache_key] = provider
        return provider

    @classmethod
    def get_kv_class(
        cls, server_args: ServerArgs, class_type: KVClassType
    ) -> Optional[Type]:
        provider = cls.get_provider(server_args)
        return provider.get_kv_class(class_type)
