from tqdm import tqdm
from pathlib import Path

import torch

from torch.utils.data import DataLoader

from transformer import Transformer, FrenchEnglishDataset, DummyDataset
from tokenizer.character_tokenizer import CharacterTokenizer
from tokenizer.bpe_tokenizer import BPETokenizer

import wandb

def main():

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="transformer",

    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": 0.03,
    #     "architecture": "Transformer",
    #     "dataset": "FrenchEnglishDataset",
    #     "epochs": 10,
    #     }
    # )

    
    tokenizer = BPETokenizer()
    # tokenizer = CharacterTokenizer()
    vocab_size = len(tokenizer.vocab)
    train_dataset = FrenchEnglishDataset(Path("smaller.csv"), tokenizer=tokenizer, train=True)
    # train_dataset = DummyDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=FrenchEnglishDataset.collate_fn)

    # val_dataset = FrenchEnglishDataset(Path("en-fr.csv"), train=False)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    model = Transformer(vocab_size=vocab_size,
                        num_layers=2,
                        num_heads=2,
                        ffn_hidden_dim=64,
                        embedding_dim=64,
                        qk_length=64,
                        value_length=64,
                        max_length=5000,
                        dropout=0.1)
    # model.load_state_dict(torch.load("ckpts/model_last.pt"))
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)

    device = "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1000):
        train_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)
        for i, (src, tgt) in train_tqdm:
            src = src.to(device)
            tgt = tgt.to(device)

            src = src.to(torch.int64)
            tgt = tgt.to(torch.int64)

            # src_dec = tokenizer.decode(src[0])
            # tgt_dec = tokenizer.decode(tgt[0])


            outputs = model(src, tgt)

            # outputs_dec = tokenizer.decode(outputs[0].argmax(dim=-1))

            # train_tqdm.set_postfix({"src": src_dec, "tgt": tgt_dec, "outputs": outputs_dec})

            # print(src_dec)
            # print(tgt_dec)
            # print(outputs_dec)

            outputs = outputs.view(-1, vocab_size)

            tgt = tgt.view(-1)

            optimizer.zero_grad()
            loss = loss_fn(outputs, tgt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_tqdm.set_postfix({"loss": loss.item()})

            # if i % 10 == 0:
            #     wandb.log({"loss": loss.item()})

            if i % 500 == 0:
                torch.save(model.state_dict(), "ckpts/model_last.pt")

        lr_scheduler.step()

        torch.save(model.state_dict(), f"ckpts/model_{epoch}_{i}.pt")

    torch.save(model.state_dict(), "ckpts/model_final.pt")

if __name__ == '__main__':
    main()
