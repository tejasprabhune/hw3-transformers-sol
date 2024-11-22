from .tokenizer import Tokenizer

import torch

class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
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
        encs = []
        for char in text:
            try:
                encs.append(self.vocab[char])
            except KeyError:
                encs.append(self.vocab[" "])
        return torch.tensor(encs)
    
    def decode(self, tokens: torch.Tensor) -> str:
        tokens = tokens.tolist()
        return "".join([self.characters[token] for token in tokens])
