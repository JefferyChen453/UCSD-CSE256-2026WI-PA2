from einops import einsum, repeat
from torch import Tensor, nn
import torch
import math

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1, device=None):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model, device=device)
        positions = torch.arange(0, max_seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe, persistent=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """Source: https://spaces.ac.cn/archives/8265/comment-page-1"""
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        self.create_positional_embedding()
    
    def create_positional_embedding(self):
        freq_list = [pow(self.theta, - (2 * (i // 2)) / self.d_k) for i in range(self.d_k)]
        freq_list = torch.tensor(freq_list).to(self.device) # (d_k, )
        pos_list = torch.arange(self.max_seq_len, device=self.device).to(self.device) # (max_seq_len,)
        cos_rot_matrix = einsum(freq_list, pos_list, "d_k, max_seq_len -> max_seq_len d_k").cos()
        sin_rot_matrix = einsum(freq_list, pos_list, "d_k, max_seq_len -> max_seq_len d_k").sin()
        R = torch.stack([cos_rot_matrix, sin_rot_matrix], dim=1) # (max_seq_len, 2, d_k)

        self.register_buffer("R", R, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # Make sure token_positions shape be like: [0, 1, 2, ...]
        if len(token_positions.shape) == 2:
            token_positions = token_positions[0]

        R = self.R[token_positions] 
        x_half1 = x[..., 0::2] # (q0 q2 q4 ...)
        x_half2 = -x[..., 1::2] # (-q1 -q3 -q5 ...)
        x_ = torch.stack([x_half2, x_half1], dim=-1).flatten(start_dim=-2) # (-q1 q0 -q3 q2 ...)
        rot_1, rot_2 = R[:, 0, :], R[:, 1, :]
        ret = x * rot_1 + x_ * rot_2

        return ret


if __name__ == "main":
    device = "cuda"
    rope = RotaryPositionalEmbedding(10000, 64, 100, device=device)
    x = torch.arange(4*50*64).view((4, 50, 64)).to(device)
    token_positions = torch.arange(50, device=x.device, dtype=torch.int32)

    print(rope(x, token_positions))