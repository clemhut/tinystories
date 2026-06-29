from torch import nn
from src.globals import D_MODEL, N_LAYERS, D_FF, N_HEADS, MAX_SEQ_LEN
from src.embedding_model.embedding_model import EmbeddingModel
from src.transformer.utils.decoder_block import DecoderBlock
import torch

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = MAX_SEQ_LEN, d_model: int = D_MODEL, n_layers: int = N_LAYERS, d_ff: int = D_FF, n_heads: int = N_HEADS, dtype = torch.float32, bias: bool = False):
        super().__init__()
        self.embedding_model = EmbeddingModel(vocab_size=vocab_size, d_model=d_model)

        self.decoder_blocks = nn.ModuleList()
        
        for i in range(n_layers):
            decoder_block = DecoderBlock(depth=i, max_seq_len=max_seq_len, d_ff=d_ff, d_model=d_model, n_heads=n_heads, bias=bias, dtype=dtype)
            self.decoder_blocks.append(decoder_block)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=bias)
        self.lm_head.weight = self.embedding_model.get_weight()  # Weight Tying

    def forward(self, X):
        """
        Input:
        X is a tensor of tokens
        X.shape =  (B, T)

        Output:
        out is a tensor of logits/probabilities for the next token
        out.shape = (B, T, vocab_size)
        """
        X = self.embedding_model.embed(X)  # (B, T, d_model)

        for block in self.decoder_blocks:
            X = block(X)  # (B, T, d_model)

        X = self.lm_head(X)  # (B, T, vocab_size)

        return X
    
    def get_param_number(self):
        return sum(parameter.numel() for parameter in self.parameters())
