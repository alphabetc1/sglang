"""Start bootstrap/kv-store-related server"""

import logging
import os
from typing import Optional, Type

from sglang.srt.disaggregation.base import BaseKVBootstrapServer
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def start_disagg_service(
    server_args: ServerArgs,
) -> Optional[BaseKVBootstrapServer]:
    """Start bootstrap server for disaggregation mode."""
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)

    if disagg_mode == DisaggregationMode.PREFILL:
        return create_bootstrap_server(
            server_args.host,
            server_args.disaggregation_bootstrap_port,
            transfer_backend,
            server_args.node_rank,
        )
    return None


def create_bootstrap_server(
    host: str,
    port: int,
    transfer_backend: TransferBackend,
    node_rank: int = 0,
) -> BaseKVBootstrapServer:
    """Create and start a bootstrap server (for dynamic mode switching)."""
    kv_bootstrap_server_class: Type[BaseKVBootstrapServer] = get_kv_class(
        transfer_backend, KVClassType.BOOTSTRAP_SERVER
    )
    bootstrap_server: BaseKVBootstrapServer = kv_bootstrap_server_class(
        host=host,
        port=port,
    )

    # Handle Ascend config store
    is_create_store = node_rank == 0 and transfer_backend == TransferBackend.ASCEND
    if is_create_store:
        try:
            from mf_adapter import create_config_store

            ascend_url = os.getenv("ASCEND_MF_STORE_URL")
            create_config_store(ascend_url)
        except Exception as e:
            error_message = (
                f"Failed create mf store, invalid ascend_url. With exception {e}"
            )
            raise RuntimeError(error_message)

    logger.info(f"Bootstrap server started on {host}:{port}")
    return bootstrap_server


def shutdown_bootstrap_server(server: BaseKVBootstrapServer) -> None:
    """Shutdown a bootstrap server."""
    if server is None:
        return
    try:
        # BaseKVBootstrapServer uses threading, call close if available
        if hasattr(server, "close"):
            server.close()
        elif hasattr(server, "shutdown"):
            server.shutdown()
        elif hasattr(server, "stop"):
            server.stop()
        logger.info("Bootstrap server stopped")
    except Exception as e:
        logger.warning(f"Error stopping bootstrap server: {e}")
