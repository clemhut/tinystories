from torch import nn, Tensor
from src.globals import D_MODEL, N_HEADS, MAX_SEQ_LEN
from src.pos_enc.positional_encoding import RoPE
import torch
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadDiffAttention(nn.Module):
    """
    MultiHead Attention module with differential attention and Grouped Query Attention
    """
    def __init__(
            self, 
            depth: int, 
            n_kv_heads: Optional[int] = None, 
            d_model: int = D_MODEL, 
            n_heads: int = N_HEADS,
            max_seq_len: int = MAX_SEQ_LEN,
            dtype = torch.float32, 
            bias: bool = False
            ):
        super().__init__()
        assert True if n_kv_heads is None else n_heads % n_kv_heads == 0, "n_kv_heads must divide n_heads"
        assert d_model % (2 * n_heads) == 0, "d_model must be divisibe by 2*n_heads"

        #### TODO: IMPLEMENT KV CACHING
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads else n_heads // 2
        self.n_rep = n_heads // self.n_kv_heads
        self.d_head = d_model // n_heads // 2 # we need to divide by 2 for differential attention
        self.dtype = dtype

        self.pos_enc = RoPE(d_head=self.d_head, max_seq_len=max_seq_len, dtype=dtype)
        self.m_q = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.m_k = nn.Linear(d_model, d_model // self.n_rep, bias=False, dtype=dtype)
        self.m_v = nn.Linear(d_model, d_model // self.n_rep, bias=False, dtype=dtype)
        self.out = nn.Linear(d_model, d_model, bias=bias, dtype=dtype)  # linear output projection

        self.lambda_init = self.lambda_init_fn(depth=depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.d_head, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.d_head, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.d_head, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.d_head, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.rmsnorm = nn.RMSNorm(2 * self.d_head)

    def lambda_init_fn(self, depth: int):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)
    
    def __repeat_kv(self, x: Tensor, n_rep: int):
        """
        Input shape: (B, H_KV, T, D_head)
        Output shape: (B, H_KV*n_rep, T, D_head)
        """
        if n_rep == 1:
            return x
        B, H_KV, T, D_head = x.shape
        return x[:, :, None, :, :].expand(B, H_KV, n_rep, T, D_head).reshape(B, H_KV*n_rep, T, D_head)
    
    def forward(self, X: Tensor):
        """
        X.shape = (B, T, D_model)
        """
        B, T, D_model = X.shape

        Q = self.m_q(X)  # (B, T, D_model)
        K = self.m_k(X)  # (B, T, D_model // n_rep)
        V = self.m_v(X)  # (B, T, D_model // n_rep)

        Q = Q.view(B, T, 2 * self.n_heads, self.d_head).transpose(1, 2)  # (B, 2*H, T, D_head)
        K = K.view(B, T, 2 * self.n_kv_heads, self.d_head).transpose(1, 2)  # (B, 2*H_KV, T, D_head)
        V = V.view(B, T, self.n_kv_heads, 2 * self.d_head).transpose(1, 2)  # (B, H_KV, T, 2*D_head)

        Q = self.pos_enc.add_pos_enc(Q)  # (B, 2*H, T, D_head)
        K = self.pos_enc.add_pos_enc(K)  # (B, 2*H_KV, T, D_head)

        K = self.__repeat_kv(K, self.n_rep)  # (B, 2*H, T, D_head)
        V = self.__repeat_kv(V, self.n_rep)  # (B, 2*H, T, 2*D_head)

        attn_weights = torch.matmul(Q, K.transpose(-1, -2))  # (B, 2*H, T, T)
        attn_weights *= self.d_head ** -0.5

        attn_mask = torch.full((T, T), float("-inf"), device=X.device, dtype=attn_weights.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)

        lambda1 = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1).float()).type_as(Q)
        lambda2 = torch.exp(torch.dot(self.lambda_q2, self.lambda_k2).float()).type_as(Q)
        lambda_full = lambda1 - lambda2 + self.lambda_init

        attn_weights = attn_weights.view(B, self.n_heads, 2, T, T)  # (B, H, 2, T, T)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]  # (B, H, T, T)

        attn = torch.matmul(attn_weights, V)  # (B, H, T, 2*D_head)
        attn = self.rmsnorm(attn)
        attn = attn * (1 - self.lambda_init)

        y = attn.transpose(1, 2).contiguous().view(B, T, D_model)  # (B, T, D_model)
        return self.out(y)
    
    
    def get_param_num(self):
        n_q_params = sum(p.numel() for p in self.m_q.parameters())
        n_k_params = sum(p.numel() for p in self.m_k.parameters())
        n_v_params = sum(p.numel() for p in self.m_v.parameters())
        n_out_params = sum(p.numel() for p in self.out.parameters())
        n_rmsnorm_params = sum(p.numel() for p in self.rmsnorm.parameters())
        return n_q_params + n_k_params + n_v_params + n_out_params + n_rmsnorm_params + self.lambda_q1.numel() + self.lambda_k1.numel() + self.lambda_q2.numel() + self.lambda_k2.numel()
