from typing import Optional

import torch
import torch.nn as nn

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
        a shape of (B, T, C) where C = num_heads * qk_length 
        (OR value_length). You are then expected to split 
        the C dimension into num_heads different heads, each 
        with shape (B, T, vec_length).

        Next, you will compute the scaled dot-product attention
        between Q, K, and V.

        Finally, you will concatenate the heads and project the
        output to have a shape of (B, T, C).

        Check out the `masked_fill` method in PyTorch to help
        you implement the masking step!
        """
        super().__init__()

        self.num_heads: int = num_heads
        self.qk_length: int = qk_length
        self.value_length: int = value_length

        # Define any layers you'll need in the forward pass
        # (hint: number of Linear layers needed != 3)
        self.W_q = nn.Linear(embedding_dim, num_heads * qk_length)
        self.W_k = nn.Linear(embedding_dim, num_heads * qk_length)
        self.W_v = nn.Linear(embedding_dim, num_heads * value_length)
        self.W_o = nn.Linear(num_heads * value_length, embedding_dim)

    def split_heads(self, x: torch.Tensor, vec_length: int) -> torch.Tensor:
        """
        Split the C dimension of the input tensor into num_heads
        different heads, each with shape (B, T, vec_length).

        Args:
            x: torch.Tensor of shape (B, T, C), where C = num_heads * vec_length
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


        if mask is not None:
            # TODO: in decoder section
            attention = lookup.masked_fill(mask == 0, float('-inf'))

        attention = torch.nn.functional.softmax(attention, dim=-1)

        print(attention.shape, V.shape)
        return torch.matmul(attention, V)


    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        attention = self.scaled_dot_product_attention(Q, K, V, mask)

        attention = self.combine_heads(attention)

        attention = self.W_o(attention)

        return attention

class FeedForwardNN(nn.Module):

    def __init__(self, 
                 embedding_dim: int,
                 hidden_dim: int):
        """
        The Feed-Forward Neural Network layer will take in
        an input tensor of shape (B, T, C) and will output
        a tensor of the same shape.

        The FFNN will have two linear layers, with a ReLU
        activation function in between.

        Args:
            hidden_dim: int, the size of the hidden layer
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Define any layers you'll need in the forward pass
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the FeedForwardNN.
        """
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x
