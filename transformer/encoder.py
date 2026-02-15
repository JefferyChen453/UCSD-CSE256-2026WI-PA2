from einops import reduce
from torch import nn
import torch.nn.functional as F

from transformer.attention import MultiHeadAttention
from transformer.positional_embedding import (
    AbsolutePositionalEmbedding,
    RotaryPositionalEmbedding,
)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, pe_type, max_seq_len=None, theta=None, device=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, device=device)
        self.ln2 = nn.LayerNorm(d_model, device=device)
        self.attn_layer = MultiHeadAttention(d_model, num_heads, pe_type, max_seq_len, theta, device)
        self.ffn_layer = nn.Sequential(
            nn.Linear(d_model, 4 * d_model).to(device),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model).to(device)
        )

    def forward(self, x, mask=None):
        x = self.ln1(x)
        attn_out, attn_score = self.attn_layer(x, mask=mask)
        x = attn_out + x
        x = self.ln2(x)
        x = self.ffn_layer(x) + x
        return x, attn_score


class TransformerEncoder(nn.Module):
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
        self.blocks = nn.ModuleList([EncoderBlock(self.d_model, self.num_heads, self.pe_type, self.max_seq_len, self.theta, self.device) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.token_embeddings(x)
        if self.pe_type == "absolute":
            x = self.positional_embedding(x)

        attn_maps = []
        for block in self.blocks:
            x, attn_map = block(x, mask=mask)
            attn_maps.append(attn_map)
        return x, attn_maps


class LinearClassifier(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, device=None):
        super().__init__()

        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.device = device

        self.linear1 = nn.Linear(d_in, d_hidden, device=device)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden, d_out, device=device)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=-1)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, num_heads, pe_type, mode="mean", max_seq_len=None, theta=None, device=None):
        super().__init__()

        self.encoder = TransformerEncoder(vocab_size, num_layers, d_model, num_heads, pe_type, max_seq_len, theta, device)
        self.classifier = LinearClassifier(d_model, 100, 3, device=device)
        self.mode = mode

    def forward(self, x, mask):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len)
        Returns:
            log_probs: (batch_size, 3)
            attn_maps: list of (batch_size, num_heads, seq_len, seq_len)
        """
        x, _ = self.encoder(x, mask=mask.unsqueeze(1).unsqueeze(2))
        if self.mode == "mean":
            x_masked = x * mask.unsqueeze(-1)
            mask_sum = mask.sum(dim=-1, keepdim=True)
            x = reduce(x_masked, "b s d -> b d", "sum")
            x /= mask_sum
        elif self.mode == "cls":
            pass
        log_probs = self.classifier(x)
        return log_probs