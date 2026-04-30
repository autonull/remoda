import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

from cache import ReMoDACache

class DepthSelectionPolicy(ABC):
    @abstractmethod
    def select_kvs(self, kv_cache: ReMoDACache, layer_idx: int, depth_slots: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Selects and returns depth KV tensors from the cache based on a specific policy.
        """
        pass

class LastNDepthPolicy(DepthSelectionPolicy):
    def select_kvs(self, kv_cache: ReMoDACache, layer_idx: int, depth_slots: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if layer_idx == 0 or len(kv_cache) == 0:
            return None, None

        slots_to_fetch = min(depth_slots, layer_idx)
        start_idx = max(0, layer_idx - slots_to_fetch)
        indices = list(range(start_idx, layer_idx))
        selected_kvs = kv_cache.get_kvs_by_indices(indices)

        if len(selected_kvs) == 0:
            return None, None

        depth_k = torch.cat([kv[0] for kv in selected_kvs], dim=2)
        depth_v = torch.cat([kv[1] for kv in selected_kvs], dim=2)

        return depth_k, depth_v

def get_depth_policy(policy_name: str) -> DepthSelectionPolicy:
    if policy_name == "last-n":
        return LastNDepthPolicy()
    else:
        # Default fallback
        return LastNDepthPolicy()
