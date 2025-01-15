import torch
import torch.nn as nn
from rmsnorm import RMSNorm
from model_args import ModelArgs
from self_attention import SelfAttention

class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the self Attention
        self.attention_norm = RMSNorm(dim = args.dim, eps = args.norm_eps)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(dim = args.dim, eps = args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):

        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex) 
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out