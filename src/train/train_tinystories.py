from src.data.tinystories_dataset import TinyStoriesDataset
from src.tokenizer.tokenizer import Tokenizer
from src.transformer.transformer import Transformer
from torch import nn
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import random


max_seq_len = 128
tokenizer = Tokenizer()
train_dataset = TinyStoriesDataset(tokenizer, block_size=max_seq_len)
val_dataset = TinyStoriesDataset(tokenizer, block_size=max_seq_len, train=False)
transformer = Transformer(vocab_size=tokenizer.get_vocab_size(), max_seq_len=max_seq_len, d_model=256, n_layers=8, n_heads=8).to(torch.device("cuda"))
checkpoint = torch.load("checkpoint.pt")
transformer.load_state_dict(checkpoint)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = AdamW(transformer.parameters(), lr=1e-5, betas=(0.9, 0.95))

def train_loop(epochs: int = 2, batch_size: int = 8):
    print(f"Transformer parameters: {transformer.get_param_number()}")
    train_losses = []

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        if epoch == 0:
            continue
        print(f"EPOCH {epoch+1}")

        # Train
        for input, labels in tqdm(train_dataloader):

            transformer.train()
            input = input.to(torch.device("cuda"))
            labels = labels.to(torch.device("cuda"))

            output = transformer(input)
            loss = loss_fn(output.view(-1, output.size(-1)), labels.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if random.random() < 0.001:
                plot_losses(train_losses, epoch)

                # Semantic Val
                transformer.eval()
                with torch.no_grad():
                    x_list = ["Once upon a time,", "Lily and", "Mom had", "Lily liked"]
                    x = random.choice(x_list)
                    x_ids = tokenizer.tokenize_to_ids(x, return_tensor=True).to(torch.device("cuda"))  # (1, T)
                    for _ in range(100):
                        logits = transformer(x_ids)  # (1, T, Vocab_size)
                        next_token_logits = logits[:, -1, :]  # (1, Vocab_size)
                        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (1, 1)
                        x_ids = torch.cat([x_ids, next_token_id], dim=1)  # (1, T+1)

                    print(f"{tokenizer.decode(x_ids[0].tolist())}")
                    torch.save(transformer.state_dict(), "checkpoint.pt")

def plot_losses(train_losses: list[int], epoch: int):
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label="train")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Train loss after {epoch+1} epochs")
    plt.grid(True)
    plt.savefig(f"losses_{epoch+1}.png")
    plt.close()

if __name__ == "__main__":
    train_loop()
