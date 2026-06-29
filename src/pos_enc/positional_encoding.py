from torch import Tensor, nn
from src.globals import D_MODEL, MAX_SEQ_LEN
import torch

class RoPE(nn.Module):
    def __init__(self, d_head: int, d_model: int = D_MODEL, max_seq_len: int = MAX_SEQ_LEN, dtype = torch.float32):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.d_head = d_head

        assert self.d_head % 2 == 0, "RoPE requires even head dimension"

        cos, sin = self.__get_cos_sin()
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)


    def add_pos_enc(self, M: Tensor) -> Tensor:
        """
        Adds RoPE positional encoding to M. M should be either Q or K
        M: (B, H, T, D_HEAD)
        """
        B, H, T, D_HEAD = M.shape
        cos = self.cos_cache[:, :, :T, :].to(device=M.device, dtype=M.dtype)
        sin = self.sin_cache[:, :, :T, :].to(device=M.device, dtype=M.dtype)
        M_rot = self.__rotate(M)

        out = M * cos + M_rot * sin

        return out
    
    def __get_cos_sin(self) -> tuple[Tensor]:
        """
        Precomputes the entires cos and sin values for the pos encodings at each positions until MAX_SEQ_LEN
        """
        d_half = self.d_head // 2
        i = torch.arange(0, d_half, dtype=torch.float32)  # (d_half)
        phi = 10000.0 ** (-2*i/self.d_head)  # (d_half)
        phi = torch.repeat_interleave(phi, repeats=2, dim=-1)  # (d_head)

        positions = torch.arange(0, self.max_seq_len, dtype=torch.float32)  # (max_seq_len)
        
        mat = torch.outer(positions, phi)  # (max_seq_len, d_head), mat[0] = [positions[0]*phi[0], ..., positions[0]*phi[d_head-1]]
        cos = torch.cos(mat)  # (max_seq_len, d_head)
        sin = torch.sin(mat)  # (max_seq_len, d_head)

        cos = cos[None, None, :, :]  # (1, 1, max_seq_len, d_head)
        sin = sin[None, None, :, :]  # (1, 1, max_seq_len, d_head)
        return (cos, sin)
    
    def __rotate(self, x: Tensor) -> Tensor:
        """
        Builds the rotated x
            --> X = [x1, x2, x3, x4] -> X_rot = [-x2, x1, -x4, x3]
        """
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x_rot
