from einops import reduce
from torch import nn
import torch
import torch.nn.functional as F

from transformer.attention import MultiHeadAttention
from transformer.positional_embedding import (
    AbsolutePositionalEmbedding,
    RotaryPositionalEmbedding,
)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, pe_type, max_seq_len=None, theta=None, device=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, device=device)
        self.ln2 = nn.LayerNorm(d_model, device=device)
        self.attn_layer = MultiHeadAttention(d_model, num_heads, pe_type, max_seq_len, theta, device)
        self.ffn_layer = nn.Sequential(
            nn.Linear(d_model, 100).to(device),
            nn.ReLU(),
            nn.Linear(100, d_model).to(device)
        )

    def forward(self, x, mask=None):
        x = self.ln1(x)
        attn_out, attn_score = self.attn_layer(x, mask=mask)
        x = attn_out + x
        x = self.ln2(x)
        x = self.ffn_layer(x) + x
        return x, attn_score


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        d_model,
        num_heads,
        pe_type,
        max_seq_len=None,
        theta=None,
        device=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.pe_type = pe_type
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device

        self.token_embeddings = nn.Embedding(vocab_size, d_model, device=device)

        if pe_type == "absolute":
            self.positional_embedding = AbsolutePositionalEmbedding(self.d_model, self.max_seq_len, dropout=0.1, device=self.device)
        elif pe_type == "rope":
            self.positional_embedding = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len, self.device)

        self.num_layers = num_layers
        self.blocks = nn.ModuleList([DecoderBlock(self.d_model, self.num_heads, self.pe_type, self.max_seq_len, self.theta, self.device) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(d_model, device=device)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, device=device)

    def forward(self, x):
        x = self.token_embeddings(x)
        if self.pe_type == "absolute":
            x = self.positional_embedding(x)

        causal_mask = torch.tril(torch.ones(x.shape[1], x.shape[1], device=self.device)).bool().unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

        attn_maps = []
        for block in self.blocks:
            x, attn_map = block(x, mask=causal_mask)
            attn_maps.append(attn_map)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits, attn_maps
