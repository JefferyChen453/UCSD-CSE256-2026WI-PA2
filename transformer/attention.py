import math

from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    attn_score = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        attn_score = attn_score.masked_fill_(~mask, -torch.inf)
    attn_score = F.softmax(attn_score, dim=-1)
    output = einsum(attn_score, V, "... queries keys, ... keys d_v -> ... queries d_v")

    return output, attn_score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, pe_type, max_seq_len=None, theta=None, device=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device

        self.q_proj_weight = torch.nn.Linear(d_model, d_model).to(device)
        self.k_proj_weight = torch.nn.Linear(d_model, d_model).to(device)
        self.v_proj_weight = torch.nn.Linear(d_model, d_model).to(device)
        self.o_proj_weight = torch.nn.Linear(d_model, d_model).to(device)
        
        self.pe_type = pe_type

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ):
        x = x.to(self.device)
        Q = self.q_proj_weight(x)
        K = self.k_proj_weight(x)
        V = self.v_proj_weight(x)

        Q = rearrange(Q, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        K = rearrange(K, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        V = rearrange(V, "... s (h d_k) -> ... h s d_k", h=self.num_heads)

        s = x.shape[-2]
        # causal_mask = torch.tril(torch.ones(s, s)).bool().to(self.device)
        if self.pe_type == "rope":
            if token_positions is None:
                token_positions = torch.arange(s, dtype=torch.int).to(self.device)
            Q = self.positional_embedding(Q, token_positions)
            K = self.positional_embedding(K, token_positions)

        output, attn_score = scaled_dot_product_attention(Q, K, V, mask)
        output = rearrange(output, "... h s d_k -> ... s (h d_k)")
        output = self.o_proj_weight(output)

        return output, attn_score