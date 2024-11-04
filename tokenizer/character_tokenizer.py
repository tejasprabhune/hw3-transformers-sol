from .tokenizer import Tokenizer

import torch

class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        We ignore capitalization.
        """
        super().__init__()

        self.vocab = {}

        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = "aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

        for i, char in enumerate(self.characters):
            self.vocab[char] = i

        if verbose:
            print("Vocabulary:", self.vocab)

    def encode(self, text: str) -> torch.Tensor:
        text = text.lower()
        return torch.tensor([self.vocab[char] for char in text])
    
    def decode(self, tokens: torch.Tensor) -> str:
        tokens = tokens.tolist()
        return "".join([self.characters[token] for token in tokens])

