import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_args import ModelArgs
from utils import apply_rotary_embedings, repeat_kv

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
        
        # Replace the entry in the cache for this token
        self.cache_k[: batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[: batch_size, start_pos : start_pos + seq_len] = xv

        # Retrieve all the cached keys and values so far
        # (B, seq_len_kv, H_KV, head_dim)
        keys = self.cache_k[: batch_size, 0 : start_pos + seq_len]
        values = self.cache_v[: batch_size, 0 : start_pos + seq_len]

        # Repeat the heads of the K & V to reach the number of heads of the queries
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, head_dim) -> (B, H_Q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, head_dim) @ (B, H_Q, head_dim, seq_len_kv) -> (B, H_Q, 1 , seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, seq_len) @ (B, H_Q, seq_len_kv, head_dim) -> (B, H_Q, 1 , head_dim) 
        output = torch.matmul(scores, values)

        # (B, H_Q, 1 , head_dim) -> (B, 1, H_Q, head_dim) -> (B, 1, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        return self.wo(output) # (B, 1, dim) -> (B, 1, dim)