from torch import nn
from src.globals import D_MODEL, N_HEADS, D_FF, MAX_SEQ_LEN
from src.transformer.utils.attention.multi_head_diff_attention import MultiHeadDiffAttention
import torch
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, depth: int, max_seq_len: int = MAX_SEQ_LEN, d_ff: int = D_FF, d_model: int = D_MODEL, n_heads: int = N_HEADS, bias: bool = False, dtype = torch.float32):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.multi_head_attention = MultiHeadDiffAttention(depth=depth, d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len, dtype=dtype)

        self.norm2 = nn.RMSNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear3 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, X):
        """
        Uses SwiGLU
        X.shape = (B, T, D_model)
        """
        X = X + self.multi_head_attention(self.norm1(X))
        residual = X
        X = self.norm2(X)
        X = residual + self.linear3(F.silu(self.linear1(X)) * self.linear2(X))
        return X
    
    def get_param_number(self):
        return sum(parameter.numel() for parameter in self.parameters())
