from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from tokenizer.character_tokenizer import CharacterTokenizer

class FrenchEnglishDataset(Dataset):
    def __init__(self, translation_csv: Path, train=True):

        self.lines = []
        self.train = train
        with open(translation_csv, "r") as file:
            file.readline()
            for line in tqdm(file):
                self.lines.append(line.strip().split("\t"))

        self.tokenizer = CharacterTokenizer()

    def __len__(self):
        if self.train:
            return int(len(self.lines) * 0.1)
        else:
            return int(len(self.lines) * 0.1)

    def __getitem__(self, idx, raw=False):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        line = self.lines[idx][0].split(",")
        english_sentence, french_sentence = line[0], line[1]

        if raw:
            return french_sentence, english_sentence

        french_encoded = self.tokenizer.encode(french_sentence)
        english_encoded = self.tokenizer.encode(english_sentence)

        return french_encoded, english_encoded

    def collate_fn(batch):
        inputs = [torch.tensor(b[0]) for b in batch]
        targets = [torch.tensor(b[1]) for b in batch]

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

        return inputs, targets
