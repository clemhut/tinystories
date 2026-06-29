from src.tokenizer.tokenizer import Tokenizer
from src.embedding_model.embedding_model import EmbeddingModel
from src.pos_enc.positional_encoding import RoPE
import torch, platform
from src.globals import N_HEADS
from src.transformer.transformer import Transformer
import torch.nn.functional as F
import time


class SkippedTest(Exception):
    pass


def build_tokenizer() -> Tokenizer:
    try:
        return Tokenizer()
    except (ModuleNotFoundError, OSError) as exc:
        raise SkippedTest(f"tokenizer is unavailable ({exc})") from exc

def tokenizer_test():
    print("TOKENIZER TESTS")
    tokenizer = build_tokenizer()
    print(tokenizer.get_bos_token())
    print(tokenizer.get_eos_token())
    print(tokenizer.get_pad_token())
    print(tokenizer.get_special_tokens())
    print(tokenizer.get_vocab_size())
    input_ids = tokenizer.tokenize_to_ids('This is a test string for the tokenizer!')
    print(input_ids)
    print(tokenizer.decode(input_ids))

def embedding_model_test():
    print("EMBEDDING MODEL TEST")
    tokenizer = build_tokenizer()
    input_tensor = tokenizer.tokenize_to_ids('This is a test string for the tokenizer!', return_tensor=True)
    print(input_tensor)
    print(input_tensor.shape)    
    embedding_model = EmbeddingModel(tokenizer.get_vocab_size())
    embeddings = embedding_model.embed(input_tensor)
    print(embeddings)
    print(embeddings.shape)

def positional_encoding_test():
    print("POSITIONAL ENCODING TEST")
    tokenizer = build_tokenizer()
    input_tensor = tokenizer.tokenize_to_ids('This is a test string for the tokenizer!', return_tensor=True)
    embedding_model = EmbeddingModel(tokenizer.get_vocab_size())
    embeddings = embedding_model.embed(input_tensor)
    
    B, T, D_model = embeddings.shape

    W_Q = torch.randn(D_model, D_model)
    W_K = torch.randn(D_model, D_model)

    Q = embeddings @ W_Q
    K = embeddings @ W_K

    D_head = D_model // N_HEADS
    Q = Q.view((B, T, N_HEADS, D_head)).transpose(1, 2)
    K = K.view((B, T, N_HEADS, D_head)).transpose(1, 2)

    positional_encoder = RoPE(d_head=D_head)
    Q_added = positional_encoder.add_pos_enc(Q)
    K_added = positional_encoder.add_pos_enc(K)

    print(Q_added.shape)
    print(K_added.shape)

def dataset_test():
    print("DATASET TEST")
    try:
        from src.data.dataset import FineWebDataset
    except ModuleNotFoundError as exc:
        print(f"SKIPPED: dataset dependencies are missing ({exc})")
        return

    try:
        dataloader = FineWebDataset.get_dataloader()
    except FileNotFoundError as exc:
        print(f"SKIPPED: dataset files are missing ({exc})")
        return

    for x in dataloader:
        print(x[0])
        print(len(x))
        print(len(x[0]))
        break

def transformer_param_number_test():
    print("TRANSFORMER PARAM NUMBER TEST")
    tokenizer = build_tokenizer()
    transformer = Transformer(vocab_size=tokenizer.get_vocab_size())
    print(transformer.get_param_number())

def transformer_forward_test():
    print("TRANSFORMER FORWARD TEST")
    if not torch.cuda.is_available():
        raise SkippedTest("CUDA is not available")

    tokenizer = build_tokenizer()
    transformer = Transformer(vocab_size=tokenizer.get_vocab_size()).to(torch.device('cuda'))

    x = "This is my little funny string, which I love and adore"
    input_ids = tokenizer.tokenize_to_ids(x, return_tensor=True).to(torch.device('cuda'))
    out = transformer(input_ids)
    print(f"Output shape: {out.shape}")
    out = out[0, -1]
    out_probs = F.softmax(out, dim=-1)
    max_idx = torch.argmax(out_probs)
    print(f"Most probable output token index and probability: {max_idx.item()}, {out_probs[max_idx]}")
    print(f"Token value: {tokenizer.decode([max_idx.item()])}")

def transformer_forward_test_cpu_vs_gpu():
    print("TRANSFORMER FORWARD TEST CPU VS GPU")
    tokenizer = build_tokenizer()
    transformer = Transformer(vocab_size=tokenizer.get_vocab_size())

    x = "This is my little funny string, which I love and adore"
    input_ids = tokenizer.tokenize_to_ids(x, return_tensor=True)

    print("CPU:")
    start = time.time()
    out = transformer(input_ids)
    elapsed = time.time() - start
    print(f"CPU took {elapsed} seconds")

    if not torch.cuda.is_available():
        raise SkippedTest("CUDA is not available")

    transformer = Transformer(vocab_size=tokenizer.get_vocab_size()).to(torch.device("cuda"))
    input_ids = tokenizer.tokenize_to_ids(x, return_tensor=True).to(torch.device("cuda"))

    print("GPU:")
    start = time.time()
    out = transformer(input_ids)
    elapsed = time.time() - start
    print(f"GPU took {elapsed} seconds")

def specs_test():
    print("SPECS TEST")
    print("platform:", platform.platform())
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    flash_available = torch.backends.cuda.is_flash_attention_available()
    print("flash built:", flash_available)
    if flash_available is False:
        print("FLASH ATTENTION MAY NOT WORK ON WINDOWS")


def main():
    test_functions = [
        tokenizer_test,
        embedding_model_test,
        positional_encoding_test,
        dataset_test,
        transformer_param_number_test,
        transformer_forward_test,
        transformer_forward_test_cpu_vs_gpu,
        specs_test,
    ]

    for index, test_fn in enumerate(test_functions):
        try:
            test_fn()
        except SkippedTest as exc:
            print(f"SKIPPED: {exc}")
        if index < len(test_functions) - 1:
            print()


if __name__ == '__main__':
    main()
