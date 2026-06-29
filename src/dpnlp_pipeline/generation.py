from __future__ import annotations

import torch


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0, device: str = "cpu") -> str:
    model = model.to(device)
    model.eval()
    input_ids = tokenizer.tokenize_to_ids(prompt, return_tensor=True).to(torch.device(device))

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    return tokenizer.decode(input_ids[0].tolist())
