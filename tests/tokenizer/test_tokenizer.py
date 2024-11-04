import unittest
from tokenizer.tokenizer import get_stats, merge_vocab

class TestTokenizer(unittest.TestCase):
    def test_get_stats(self):
        """Test basic `get_stats` example."""
        self.assertEqual(get_stats([1, 2, 3, 1, 2]), {(1, 2): 2, (2, 3): 1, (3, 1): 1})

    def test_empty_ids(self):
        """Test empty input."""
        self.assertEqual(get_stats([]), {})

    def test_merge_vocab(self):
        """Test basic `merge_vocab` example."""
        self.assertEqual(merge_vocab([1, 2, 3, 1, 2], (1, 2), 4), [4, 3, 4])

    def test_merge_vocab_no_pair(self):
        """Test `merge_vocab` with no pair to merge."""
        self.assertEqual(merge_vocab([1, 2, 3, 1, 2], (1, 3), 4), [1, 2, 3, 1, 2])

    def test_merge_vocab_empty_ids(self):
        """Test `merge_vocab` with empty input."""
        self.assertEqual(merge_vocab([], (1, 2), 3), [])

if __name__ == '__main__':
    unittest.main()
