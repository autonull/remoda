from dataclasses import dataclass

@dataclass
class ReMoDAConfig:
    vocab_size: int = 50257
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    max_position_embeddings: int = 1024

    # Ablation Toggles
    use_rt_kv: bool = True               # True = ReMoDA/RT (project KV from output), False = Std Transformer (project KV from input)
    use_moda: bool = True                # True = ReMoDA/MoDA (cross-layer depth attention), False = Std Self-Attention
    depth_selection_policy: str = "last-n"  # "last-n" or "last-n+random-1"
    norm_placement: str = "post-norm"    # "pre-norm" or "post-norm"
    depth_slots: int = 1                 # How many historical layers to retrieve
