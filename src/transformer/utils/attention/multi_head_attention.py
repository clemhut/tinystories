from torch import nn, Tensor
from src.globals import D_MODEL, N_HEADS
from src.pos_enc.positional_encoding import RoPE
import torch
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    MultiHead Attention module
    """
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, dtype = torch.float32, bias: bool = False):
        super().__init__()
        self.__n_heads = n_heads
        self.__d_head = d_model // n_heads

        self.pos_enc = RoPE(d_model=d_model, n_heads=n_heads, dtype=dtype)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False, dtype=dtype)  # returns [Q, K, V]
        self.out = nn.Linear(d_model, d_model, bias=bias, dtype=dtype)  # linear output projection

    def forward(self, X: Tensor):
        """
        X.shape = (B, T, D_model)
        """
        B, T, D_model = X.shape

        qkv = self.qkv(X)  # (B, T, D_model*3)
        Q, K, V = qkv.chunk(3, dim=-1)  # each is (B, T, D_model)

        Q = Q.view(B, T, self.__n_heads, self.__d_head).transpose(1, 2)  # (B, H, T, D_head)
        K = K.view(B, T, self.__n_heads, self.__d_head).transpose(1, 2)  # (B, H, T, D_head)
        V = V.view(B, T, self.__n_heads, self.__d_head).transpose(1, 2)  # (B, H, T, D_head)

        Q = self.pos_enc.add_pos_enc(Q)  # (B, H, T, D_head)
        K = self.pos_enc.add_pos_enc(K)  # (B, H, T, D_head)

        y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)  # (B, H, T, D_head)
        y = y.transpose(1, 2).contiguous().view(B, T, D_model)  # (B, T, D_model)
        return self.out(y)
    
    def get_param_num(self):
        n_qkv_params = sum(p.numel() for p in self.qkv.parameters())
        n_out_params = sum(p.numel() for p in self.out.parameters())
        return n_qkv_params + n_out_params
