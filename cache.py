import torch
from typing import Tuple, List, Optional

class ReMoDACache:
    """
    A simple cache mechanism for storing and retrieving KV states
    across layers for the ReMoDA architecture.
    """
    def __init__(self):
        self._k_cache: List[torch.Tensor] = []
        self._v_cache: List[torch.Tensor] = []

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> None:
        """Stores the KV states for the given layer."""
        if layer_idx < 0:
            raise ValueError(f"Invalid layer index: {layer_idx}")

        if layer_idx >= len(self._k_cache):
            # Pad with None if we skipped layers
            for _ in range(layer_idx - len(self._k_cache)):
                self._k_cache.append(None)
                self._v_cache.append(None)
            self._k_cache.append(key_states)
            self._v_cache.append(value_states)
        else:
            self._k_cache[layer_idx] = key_states
            self._v_cache[layer_idx] = value_states

    def __len__(self) -> int:
        return len(self._k_cache)

    def get_kvs_by_indices(self, indices: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        selected = []
        for i in indices:
            if i >= 0 and i < len(self._k_cache):
                k = self._k_cache[i]
                v = self._v_cache[i]
                if k is not None and v is not None:
                    selected.append((k, v))
        return selected

    def get_all_kvs(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return list(zip(self._k_cache, self._v_cache))
