from src.tokenizer.tokenizer import Tokenizer
from src.transformer.transformer import Transformer
import torch

max_seq_len = 512
tokenizer = Tokenizer()
transformer = Transformer(vocab_size=tokenizer.get_vocab_size(), max_seq_len=max_seq_len, d_model=512, n_layers=8, n_heads=8).to(torch.device("cuda"))
checkpoint = torch.load("checkpoint_2.pt")
transformer.load_state_dict(checkpoint)
temperature = 0.6

print(transformer.get_param_number())
print(f"WELCOME TO THE CHATBOT! Temperature={temperature}")
while True:
    prompt = input("Write your prompt or 'change_temperature VAL': ")  # (text completion)

    if ("change_temperature" in prompt):
        temperature = float(prompt.split(" ")[1])
        print(f"Temperature changed to {temperature}")
        continue

    prompt_ids = tokenizer.tokenize_to_ids(prompt, return_tensor=True).to(torch.device("cuda"))  # (1, T)
    iters = max_seq_len - prompt_ids.shape[1]
    
    for _ in range(iters):
        with torch.no_grad():
            logits = transformer(prompt_ids)  # (1, T, vocab_size)
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)
            next_token_logits /= temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
            #next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (1, 1)
        prompt_ids = torch.cat([prompt_ids, next_token_id], dim=-1)  # (1, T+1)

        new_text = tokenizer.decode(next_token_id[0].tolist())
        print(new_text, end="", flush=True)
        if (new_text == tokenizer.get_eos_token()):
            break
    print()
