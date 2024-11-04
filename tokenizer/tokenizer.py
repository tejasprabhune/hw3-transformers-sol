from typing import List, Dict, Tuple, Optional

# Inspired by https://github.com/karpathy/minbpe and
# Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2015)
# arXiv:1508.07909

def get_stats(ids: List[int], counts: Optional[Dict[Tuple, int]] = None) -> Dict[Tuple, int]:
    """
    Returns a dictionary with the count of each unique pair of consecutive
    elements in the input list.

    Example:
        [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for i in range(1, len(ids)):
        pair = (ids[i - 1], ids[i])
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_vocab(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """
    Merges the pair of elements in the input list with the new id.

    Example:
        [1, 2, 3, 1, 2], (1, 2), 4 -> [4, 3, 4]
    """
    new_ids = []
    i = 0
    while i < len(ids):
        # If we have space (i.e. not at the very end) and the pair we find matches the input pair,
        # we replace the pair with the new id.
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(new_id)

            # Increment by 2 since we process a pair.
            i += 2
        else:
            # Otherwise, we just append the current element and move to the next one.
            new_ids.append(ids[i])
            i += 1
    return new_ids

class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError
    
    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError
