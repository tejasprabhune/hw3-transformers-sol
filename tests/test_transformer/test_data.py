from transformer import FrenchEnglishDataset

import unittest

class TestFrenchEnglishDataset(unittest.TestCase):
    dataset = FrenchEnglishDataset("en-fr-full.csv")

    def test_len(self):
        """Test that the dataset has the correct length."""
        self.assertEqual(len(self.dataset), 2)

    def test_get_item(self):
        """Test that we can retrieve an item from the dataset."""

        french_encoded, english_encoded = self.dataset[123789]

        self.assertTrue(len(french_encoded) > 0)
