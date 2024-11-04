from typing import Optional

import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, 
                 vocab_size: int, 
                 num_layers: int, 
                 num_heads: int,
                 qk_length: int,
                 value_length: int):
        """
        Remember that the encoder will take in a sequence
        of tokens and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Encoder layers,
        each of which will have a Multi-Head Attention layer
        and a Feed-Forward Neural Network layer. For this, we
        need to specify the number of layers and the number of
        heads.

        Additionally, for every Multi-Head Attention layer, we
        need to know how long each query/key is, and how long
        each value is.
        """

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass


    def forward(self):
        pass

class EncoderLayer(nn.Module):
    
    def __init__(self, 
                 num_heads: int, 
                 qk_length: int, 
                 value_length: int):
        """
        Each encoder layer will take in an embedding of
        shape (B, T, C) and will output an encoded representation
        of the same shape.

        The encoder layer will have a Multi-Head Attention layer
        and a Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding!
        """

        self.num_heads = num_heads
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
    
    
    def forward(self):
        pass

class MultiHeadAttention(nn.Module):

    def __init__(self, 
                 num_heads: int,
                 embedding_dim: int,
                 qk_length: int,
                 value_length: int
                 ):
        """
        The Multi-Head Attention layer will take in Q, K, and V
        matrices and will output an attention matrix of shape <TODO>.

        First, Q, K, and V should be projected to have
        a shape of (B, T, C) where C = num_heads * qk_length. You are
        then expected to split the C dimension into num_heads
        different heads, each with shape (B, T, qk_length).

        Next, you will compute the scaled dot-product attention
        between Q, K, and V.

        Finally, you will concatenate the heads and project the
        output to have a shape of (B, T, C).

        Check out the `masked_fill` method in PyTorch to help
        you implement the masking step!
        """
        super().__init__()

        self.num_heads = num_heads
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # (hint: number of Linear layers needed != 3)
        self.W_q = nn.Linear(embedding_dim, num_heads * qk_length)
        self.W_k = nn.Linear(embedding_dim, num_heads * qk_length)
        self.W_v = nn.Linear(embedding_dim, num_heads * value_length)
        self.W_o = nn.Linear(num_heads * value_length, embedding_dim)

    def split_heads(self, x: torch.Tensor, vec_length: int) -> torch.Tensor:
        """
        Split the C dimension of the input tensor into num_heads
        different heads, each with shape (B, T, qk_length).

        Args:
            x: torch.Tensor of shape (B, T, C), where C = num_heads * qk_length
            vec_length: int, the length of the query/key/value vectors

        Returns:
            torch.Tensor of shape (B, num_heads, T, vec_length)
        """
        
        B, T, C = x.size()

        assert C // self.num_heads == vec_length, \
            "Input tensor does not have the correct shape for splitting."


        x = x.view(B, T, self.num_heads, vec_length)

        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the num_heads different heads into a single tensor.
        Hint: check out the `contiguous` method in PyTorch to help
        you reshape the tensor.

        Args:
            x: torch.Tensor of shape (B, num_heads, T, vec_length)

        Returns:
            torch.Tensor of shape (B, T, num_heads * vec_length)
        """
        
        B, num_heads, T, vec_length = x.size()

        x = x.permute(0, 2, 1, 3)

        return x.contiguous().view(B, T, num_heads * vec_length)
    

    def scaled_dot_product_attention(self, 
                                     Q: torch.Tensor, 
                                     K: torch.Tensor, 
                                     V: torch.Tensor, 
                                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the scaled dot-product attention given Q, K, and V.

        Args:
            Q: torch.Tensor of shape (B, num_heads, T, qk_length)
            K: torch.Tensor of shape (B, num_heads, T, qk_length)
            V: torch.Tensor of shape (B, num_heads, T, value_length)
            mask: Optional torch.Tensor of shape (B, T, T) or None
        """
        lookup = torch.matmul(Q, K.transpose(-2, -1)) 
        lookup = lookup / torch.sqrt(torch.tensor(Q.size(-1)))

        attention = torch.nn.functional.softmax(lookup, dim=-1)

        if mask is not None:
            # TODO: in decoder section
            attention = attention.masked_fill(mask == 0, -1e9)

        return torch.matmul(attention, V)


    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Multi-Head Attention layer.

        Args:
            Q: torch.Tensor of shape (B, T, C)
            K: torch.Tensor of shape (B, T, C)
            V: torch.Tensor of shape (B, T, C)

        Returns:
            torch.Tensor of shape (B, T, C)
        """

        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_heads(Q, self.qk_length)
        K = self.split_heads(K, self.qk_length)
        V = self.split_heads(V, self.value_length)

        attention = self.scaled_dot_product_attention(Q, K, V)

        attention = self.combine_heads(attention)

        attention = self.W_o(attention)

        return attention
