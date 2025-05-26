import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention module for temporal sequences
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Check if hidden size is divisible by number of heads
        assert self.head_dim * num_heads == hidden_size, "Hidden size must be divisible by number of heads"

        # Linear projections for query, key and value
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        # Dropout for attention
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for dot product attention
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        """
        Forward pass for temporal self-attention

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Attention output of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.size()

        # Project inputs to queries, keys and values
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Reshape for multi-head attention
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply attention mask
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute attention output
        # [batch_size, num_heads, seq_len, head_dim]
        attention_output = torch.matmul(attention_weights, v)

        # Transpose back to [batch_size, seq_len, num_heads, head_dim]
        attention_output = attention_output.transpose(1, 2)

        # Reshape back to [batch_size, seq_len, hidden_size]
        attention_output = attention_output.contiguous().view(batch_size, seq_len, self.hidden_size)

        # Final projection
        output = self.output_proj(attention_output)

        return output