import torch 
import torch.nn as nn 
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional 


@dataclass 
class ModelArgs:
    dim: int=4096
    n_layers: int=32
    n_q_heads: int=32
    n_kv_heads: int=None
    vocab_size: int=-1
    multiple_of: int=256
    ffn_dim_multiplier: float=None
    norm_eps: float=1e-5

    # For KV cache
    max_batch_size = 32
    max_seg_len = 2048

    device: str = None


class RMSnorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1))

    def _rms_norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):      
        # x: (Batch, Seq_len, Dim)
        # (Dim) * (Bacth, Seq_len, Dim ) -> (Bacth, Seq_len, Dim )
        return self.weight * self._rms_norm(x.float()).as_type(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len:int, device: str, theta: int=10000):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # (Head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2)
    # (Head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator/head_dim)).to(device)
    # (Seq_len)
    m = torch.arange(seq_len, device=device)

    # (Seq_len, Head_dim / 2)
    matrix = torch.outer(m, theta)
    return torch.polar(torch.ones_like(matrix, matrix))


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # x: (Batch, Seq_len, Head, Head_dim)
    # freqs_complex: (Seq_len, Head_dim / 2)

    # (Batch, Seq_len, Head, Head_dim) -> (Batch, Seq_len, Head, Head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # (Seq_len, Head_dim / 2) -> (1, Seq_len, 1, Head_dim / 2)
    freq_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # (Batch, Seq_len, Head, Head_dim / 2) * (1, Seq_len, 1, Head_dim / 2) -> (Batch, Seq_len, Head, Head_dim / 2)
    x_rotated = x_complex * freq_complex

    # (Batch, Seq_len, Head, Head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)

    # # (Batch, Seq_len, Head, Head_dim)
    x_out = x_out(*x.shape)

    return x_out.as_type(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int):
    # x: (Batch, 1, Q_head, Head_dim)
    batch, seq_len, head, head_dim = x.shape

    if n_rep == 1:
        return x
    else:
        x = x[:, :, :, None, :].expand(
            batch, seq_len, head, 2, head_dim
        ).reshape(
            batch, seq_len, head*2, head_dim
        )
        return x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_q_heads = args.n_q_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim / self.n_q_heads
        self.n_rep = self.n_q_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, self.n_q_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(self.n_q_heads * args.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, self.max_seg_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, self.max_seg_len, self.n_kv_heads, self.head_dim))

    def forward(
            self, 
            x: torch.Tensor,
            freqs_complex: torch.Tensor,
            start_pos: int
        ):
        # (Batch, Seq_len, Dim)
        batch, seq_len, _ = x.shape

        # (Batch, 1, Dim) -> (Batch, 1, Q_head * Head_dim)
        self.xq = self.wq(x)
        # (Batch, 1, Dim) -> (Batch, 1, KV_head * Head_dim)
        self.xk = self.wk(x)
        # (Batch, 1, Dim) -> (Batch, 1, KV_head * Head_dim)
        self.xv = self.wv(x)

        # (Batch, 1, Q_head * Head_dim) -> (Batch, 1, Q_head, Head_dim)
        xq = xq.view((batch, seq_len, self.n_q_head, self.head_dim))
        # (Batch, 1, KV_head * Head_dim) -> (Batch, 1, KV_head, Head_dim)
        xk = xk.view((batch, seq_len, self.n_kv_head, self.head_dim))
        # (Batch, 1, KV_head * Head_dim) -> (Batch, 1, KV_head, Head_dim)
        xv = xv.view((batch, seq_len, self.n_kv_head, self.head_dim))

        # (Batch, 1, Q_head, Head_dim) -> (Batch, 1, Q_head, Head_dim)
        xq = apply_rotary_embeddings(xq, freqs_complex)
        # (Batch, 1, KV_head, Head_dim) -> (Batch, 1, KV_head, Head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex)

        # Save to cache
        self.cache_k[:batch, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch, start_pos:start_pos+seq_len] = xv

        # Get keys, values
        # (Batch, KV_seq_len, KV_head, Head_dim)
        keys = self.cache_k[:batch, :start_pos+seq_len]
        # (Batch, KV_seq_len, KV_head, Head_dim)
        values = self.cache_v[:batch, :start_pos+seq_len]

        # Repeat
        # (Batch, KV_seq_len, KV_head, Head_dim) -> (Batch, KV_seq_len, Q_head, Head_dim)
        keys = repeat_kv(keys, self.n_rep)
        # (Batch, KV_seq_len, KV_head, Head_dim) -> (Batch, KV_seq_len, Q_head, Head_dim)
        values = repeat_kv(values, self.n_rep)

        # Transpose
        # (Batch, 1, Q_head, Head_dim) -> (Batch, Q_head, 1, Head_dim)
        xq = xq.transpose(1, 2)
        # (Batch, KV_seq_len, Q_head, Head_dim) -> (Batch, Q_head, KV_seq_len, Head_dim)
        keys = keys.transpose(1, 2)
        # (Batch, KV_seq_len, Q_head, Head_dim) -> (Batch, Q_head, KV_seq_len, Head_dim)
        values = values.transpose(1, 2)

        # Score
        # (Batch, Q_head, 1, Head_dim) @ (Batch, Q_head, Head_dim, KV_seq_len) -> (Batch, Q_head, 1, KV_seq_len)
        score = xq @ keys.transpose(2, 3) / self.head_dim ** 0.5
        score = F.softmax(score.float(), dim=-1).as_type(xq)

        # Output
        # (Batch, Q_head, 1, KV_seq_len) @ (Batch, Q_head, KV_seq_len, Head_dim) -> (Batch, Q_head, 1, Head_dim)
        output = score @ values

        # Transpose
        # (Batch, Q_head, 1, Head_dim) -> (Batch, 1, Dim)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return output


class SwiGLU_Feed_Forward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim 
        hidden_dim = (2 * hidden_dim) / 3

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        hidden_dim = args.multiple_of * ((hidden_dim + (args.multiple_of - 1)) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (Batch, Seq_len, Dim) -> (Batch, Seq_len, Hidden_dim)
        swiglu = F.silu(self.w1(x))

        # (Batch, Seq_len, Dim) -> (Batch, Seq_len, Hidden_dim)
        x_v = self.w3(x)

        # (Batch, Seq_len, Hidden_dim) -> (Batch, Seq_len, Hidden_dim)
        x = swiglu*x_v

        # (Batch, Seq_len, Hidden_dim) -> (Batch, Seq_len, Dim)
        x = self.w2(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Multi-head attention
        self.attention_norm = RMSnorm(args.dim, eps=args.norm_eps)
        self.attention = Attention()
        
        # Fead Forward 
        self.ffn_norm = RMSnorm(args.dim, eps=args.norm_eps)
        self.ffn = SwiGLU_Feed_Forward()
    
    def forward(self, x: torch.Tensor):
        # x: (Batch, Seq_len, Dim)

        # (Batch, Seq_len, Dim) -> (Batch, Seq_len, Dim) (Batch, Q_head, 1, Head_dim)
        h = x + self.attention(self.attention_norm(x))
        # (Batch, Seq_len, Dim) -> (Batch, Seq_len, Dim)
        h = h + self.ffn(self.ffn_norm(x))

        return h


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size==-1, "Vocab size must be set"

        self.args = args
        self.n_layers = args.n_layers
        self.token_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList
        for _ in range(args.layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSnorm(args.dim, eps=args.norm_eps)
        self.feqs_complex = precompute_theta_pos_frequencies()

        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int):
        # x: (Batch, Seq_len) -> (Batch, Seq_len, Dim)
        batch, seg_len = x.shape
        
        # (Batch, Seq_len) -> (Batch, Seq_len, Dim)
        h = self.token_embeddings(x)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seg_len]

        for layer in self.layers:
            h = layer()

        # Normalize
        h = self.norm(h)

        # Output
        h = self.output(h)

        return h