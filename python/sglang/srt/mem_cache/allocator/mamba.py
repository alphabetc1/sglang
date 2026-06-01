"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MambaPool


class MambaTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Slot allocator for the Mamba state pool.

    Mamba caches one whole state tensor per request, so the allocator hands out
    fixed-size slots (1 per request) rather than paged token KV indices. The
    underlying tensor storage lives in `MambaPool`; this class owns only the
    free-slot bookkeeping.
    """

    def __init__(self, size: int, device: str, kvcache: "MambaPool"):
        # Mamba has no paging, no dtype on the allocator side, and no need to
        # sort released slots — skip the base __init__ signature and set the
        # fields that base helpers rely on by hand.
        self.size = size
        self.page_size = 1
        self.device = device
        self._kvcache = kvcache
        self.need_sort = False

        # Unused — kept so base helpers that reference them do not AttributeError.
        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

        self.clear()

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.free_slots = torch.cat((self.free_slots, free_index))
        else:
            self.free_group.append(free_index)

    def clear(self):
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []

    def backup_state(self):
        return self.free_slots

    def restore_state(self, state):
        self.free_slots = state

    def clear_slots(self, indices: torch.Tensor):
        """Zero out cached mamba state at the given slot indices."""
        self._kvcache.clear_slots(indices)
