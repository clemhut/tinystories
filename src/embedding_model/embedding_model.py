from torch import nn
from src.globals import D_MODEL
from torch import Tensor

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = D_MODEL):
        super().__init__()
        self.__d_model = d_model
        self.__vocab_size = vocab_size
        self.__embedding = nn.Embedding(vocab_size, d_model)

    def embed(self, ids: Tensor) -> Tensor:
        return self.__embedding(ids)
    
    def get_d_model(self) -> int:
        return self.__d_model
    
    def get_vocab_size(self) -> int:
        return self.__vocab_size
    
    def get_weight(self) -> Tensor:
        return self.__embedding.weight
    
    def get_param_number(self):
        return sum(p.numel() for p in self.__embedding.parameters())
    
