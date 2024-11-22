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
            return int(len(self.lines) * 1.0)
        else:
            return int(len(self.lines) * 0.2)

    def __getitem__(self, idx, raw=False):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        line = self.lines[idx][0].split(",")
        english_sentence, french_sentence = line[0], line[1]
        print(english_sentence, french_sentence)

        if raw:
            return french_sentence, english_sentence

        french_encoded = self.tokenizer.encode(french_sentence)
        english_encoded = self.tokenizer.encode(english_sentence)

        return french_encoded, english_encoded

