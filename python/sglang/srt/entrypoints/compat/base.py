from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from fastapi import Request
from fastapi.responses import Response


@dataclass(frozen=True)
class CompatRouteSpec:
    path: str
    methods: Tuple[str, ...] = ("POST",)
    include_in_schema: bool = False


@dataclass
class CompatServiceRegistry:
    """Core services that compatibility adapters can reuse."""

    rerank: Any = None


class CompatAdapter(ABC):
    """Base class for external-protocol adapters."""

    name: str

    @abstractmethod
    def route_specs(self) -> Iterable[CompatRouteSpec]:
        """Return the route specs exposed by this adapter."""

    @abstractmethod
    async def handle_request(
        self,
        raw_request: Request,
        services: CompatServiceRegistry,
    ) -> Response:
        """Handle an incoming request using the provided core services."""
