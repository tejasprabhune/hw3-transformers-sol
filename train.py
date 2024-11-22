from tqdm import tqdm
from pathlib import Path

import torch

from torch.utils.data import DataLoader

from transformer import Transformer, FrenchEnglishDataset
from tokenizer.character_tokenizer import CharacterTokenizer

def main():

    train_dataset = FrenchEnglishDataset(Path("en-fr-full.csv"), train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=FrenchEnglishDataset.collate_fn)

    # val_dataset = FrenchEnglishDataset(Path("en-fr.csv"), train=False)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    tokenizer = CharacterTokenizer()
    vocab_size = len(tokenizer.vocab)

    model = Transformer(vocab_size=vocab_size,
                        num_layers=6,
                        num_heads=8,
                        ffn_hidden_dim=512,
                        embedding_dim=256,
                        qk_length=256,
                        value_length=256,
                        max_length=5000,
                        dropout=0.1)

    device = 0
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(10):

        train_tqdm = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (src, tgt) in train_tqdm:
            src = src.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()
            outputs = model(src, tgt)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, vocab_size), tgt.view(-1))
            loss.backward()
            optimizer.step()

            train_tqdm.set_postfix({"loss": loss.item()})

if __name__ == '__main__':
    main()
