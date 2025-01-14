import torch
import torch.nn as nn
from model_args import ModelArgs
from utils import apply_rotary_embedings

class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates the number of heads for the key and value
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries: 
        self.n_head_q = args.n_heads
        # Indicates howw many times the heads of Keys and Values should be repeated to match the head of the Queries
        self.n_rep = self.n_head_q // self.n_kv_heads
        # Indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):

        batch_size, seq_len, _ = x.shape # (B, 1, dim)
        
        # Apply the Wq, Wk and Wv matrices to queries, keys & values 
        # (B, 1, dim) -> (B, 1, H_Q * head_dim)
        xq = self.wq(x)
        # (B, 1, dim) -> (B, 1, H_KV * head_dim)
        xv = self.wv(x)
        xk = self.wk(x)

        # (B, 1, H_Q * head_dim) -> (B, 1, H_Q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_head_q, self.head_dim)
        # (B, 1, H_KV * head_dim) -> (B, 1, H_KV, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Does not change the shape of the vectors
        xq = apply_rotary_embedings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embedings(xk, freqs_complex, device=x.device)
        
