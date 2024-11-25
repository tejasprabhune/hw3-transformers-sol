import unittest
from tokenizer.bpe_tokenizer import BPETokenizer

class TestBPETokenizer(unittest.TestCase):
    def test_e2e(self):
        """Test basic end-to-end example."""
        tokenizer = BPETokenizer()
        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, "hello")

    def test_empty_input(self):
        """Test empty input."""
        tokenizer = BPETokenizer()
        encoded = tokenizer.encode("")
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, "")

    def test_french_input(self):
        """Test example from French-English dataset."""
        tokenizer = BPETokenizer()
        text = ("Souvent considérée comme la plus ancienne des sciences, "
                 "elle découle de notre étonnement et de nos q...")
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_english_input(self):
        """Test example from English-French dataset."""
        tokenizer = BPETokenizer()
        text = ("The white light spectrum Codes in the light"
                " The electromagnetic spectrum Emission spectra Absorption...")
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_special_char(self):
        tokenizer = BPETokenizer()
        text = ("© 2006 An original idea and a realization of the ASTROLab of Mont-Mégantic National Park")
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

if __name__ == '__main__':
    unittest.main()
