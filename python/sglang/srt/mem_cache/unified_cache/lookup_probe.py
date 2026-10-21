"""Throwaway probe: per-call timings of the external linker lookup.

Enabled by LINKER_LOOKUP_PROBE_DIR; writes one CSV per process.
"""

import os
import threading
import time

_DIR = os.environ.get("LINKER_LOOKUP_PROBE_DIR")
_lock = threading.Lock()
_files = {}


def enabled() -> bool:
    return _DIR is not None


def record(kind: str, *fields) -> None:
    if _DIR is None:
        return
    pid = os.getpid()
    with _lock:
        f = _files.get(pid)
        if f is None:
            os.makedirs(_DIR, exist_ok=True)
            f = open(os.path.join(_DIR, f"probe_{pid}.csv"), "a", buffering=1 << 16)
            _files[pid] = f
        f.write(
            ",".join(
                [kind, f"{time.time():.6f}", threading.current_thread().name]
                + [str(x) for x in fields]
            )
            + "\n"
        )
