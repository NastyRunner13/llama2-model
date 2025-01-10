from typing import Optional
from dataclasses import dataclass

@dataclass 
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the queries
    n_kv_heads: Optional[int] = None # Number of heads for the K & V
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # For KV Cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None