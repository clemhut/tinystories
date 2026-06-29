from transformers import AutoTokenizer
from torch import Tensor

class Tokenizer:
    def __init__(self):
        self.__tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")

    def tokenize_to_ids(self, text: str, return_tensor=False) -> list[int] | Tensor:
        if return_tensor:
            return self.__tokenizer(text, truncation=False, return_attention_mask=False, return_tensors="pt")["input_ids"]
        else:
            return self.__tokenizer(text, truncation=False, return_attention_mask=False)["input_ids"]
    
    def decode(self, input_ids: list[int]) -> str:
        return self.__tokenizer.decode(input_ids)
    
    def get_vocab_size(self) -> int:
        return self.__tokenizer.vocab_size
    
    def get_special_tokens(self) -> dict:
        return self.__tokenizer.special_tokens_map
    
    def get_eos_token(self) -> str:
        return self.__tokenizer.eos_token
    
    def get_bos_token(self) -> str:
        return self.__tokenizer.bos_token
    
    def get_pad_token(self) -> str:
        return self.__tokenizer.pad_token
    
    def get_pad_token_id(self) -> int:
        return self.__tokenizer.pad_token_id