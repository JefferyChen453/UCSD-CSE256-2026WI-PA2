import math

from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
import torch
import torch.nn.functional as F


def _make_relative_position_ids(seq_len: int, max_relative_pos: int, device: torch.device) -> Int[Tensor, "seq_len seq_len"]:
    """Relative position id for (i, j): j - i, clamped to [-max_relative_pos, max_relative_pos]. Returns indices in [0, 2*max_relative_pos]."""
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    rel_pos = (j - i).clamp(-max_relative_pos, max_relative_pos) + max_relative_pos
    return rel_pos


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    attn_score = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        mask = mask.bool() if mask.dtype != torch.bool else mask
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
        pe_type: str = None,
        positional_embedding: nn.Module = None,
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
            Q = positional_embedding(Q, token_positions)
            K = positional_embedding(K, token_positions)

        output, attn_score = scaled_dot_product_attention(Q, K, V, mask)
        output = rearrange(output, "... h s d_k -> ... s (h d_k)")
        output = self.o_proj_weight(output)

        return output, attn_score


class DisentangledMultiHeadAttention(nn.Module):
    """
    DeBERTa-style disentangled attention: content-to-content + content-to-position + position-to-content.
    No absolute positional encoding on input; uses relative position embeddings only.
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.max_relative_pos = max_seq_len
        self.num_relative_pos = 2 * self.max_relative_pos + 1
        self.device = device

        self.q_proj = nn.Linear(d_model, d_model, device=device)
        self.k_proj = nn.Linear(d_model, d_model, device=device)
        self.v_proj = nn.Linear(d_model, d_model, device=device)
        self.o_proj = nn.Linear(d_model, d_model, device=device)

        self.rel_embed = nn.Embedding(self.num_relative_pos, d_model, device=device)
        self.pos_key_proj = nn.Linear(d_model, d_model, device=device)
        self.pos_query_proj = nn.Linear(d_model, d_model, device=device)

        self.scale_factor = 3.0

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        mask: Bool[Tensor, "batch 1 1 seq_len"] | None = None,
    ):
        x = x.to(self.device)
        b, s, _ = x.shape
        Q = rearrange(self.q_proj(x), "b s (h d_k) -> b h s d_k", h=self.num_heads)
        K = rearrange(self.k_proj(x), "b s (h d_k) -> b h s d_k", h=self.num_heads)
        V = rearrange(self.v_proj(x), "b s (h d_k) -> b h s d_k", h=self.num_heads)

        scale = 1.0 / math.sqrt(self.d_k * self.scale_factor)
        c2c = einsum(Q, K, "b h q d, b h k d -> b h q k") * scale

        rel_idx_c2p = _make_relative_position_ids(s, self.max_relative_pos, x.device)
        rel_idx_p2c = 2 * self.max_relative_pos - rel_idx_c2p

        rel_emb_c2p = self.rel_embed(rel_idx_c2p)
        rel_emb_p2c = self.rel_embed(rel_idx_p2c)
        pos_key = rearrange(
            self.pos_key_proj(rel_emb_c2p), "q k (h d_k) -> h q k d_k", h=self.num_heads
        )
        pos_query = rearrange(
            self.pos_query_proj(rel_emb_p2c), "q k (h d_k) -> h q k d_k", h=self.num_heads
        )

        c2p = einsum(Q, pos_key, "b h q d, h q k d -> b h q k") * scale
        p2c = einsum(pos_query, K, "h q k d, b h k d -> b h q k") * scale

        attn_score = c2c + c2p + p2c
        if mask is not None:
            mask = mask.bool() if mask.dtype != torch.bool else mask
            attn_score = attn_score.masked_fill_(~mask, -torch.inf)
        attn_weights = F.softmax(attn_score, dim=-1)
        out = einsum(attn_weights, V, "b h q k, b h k d -> b h q d")
        out = rearrange(out, "b h s d_k -> b s (h d_k)")
        out = self.o_proj(out)
        return out, attn_weights