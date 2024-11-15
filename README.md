# HW 3: Transformers!

In this HW, you will implement the Transformer architecture from scratch.
You'll use all the knowledge and wisdom you've accumulated over the
last 2-3 weeks and finally apply it to build this model.

Since this HW is long (and a bit unfinished), we are breaking it down
into a few steps.

1. Implement tokenizers, Multi-Head Attention, Scaled Dot-Product Attention, and
   Feed-Forward Networks.
2. Implement the Encoder and Decoder layers.
3. Implement the Encoder and Decoder stacks.
4. Implement the Transformer model.
5. Train the model on a simple task (tbd!).

You can get started by exploring the code!

We've provided a few sanity checks:
`python -m unittest tests/tokenizer/test_character_tokenizer.py`
`python -m unittest tests/transformer/test_attention.py`

However, these are not exhaustive, and also do not check for
correctness of the model. Rather, they check that you are
correctly outputting the expected shapes and types.

Good luck!
