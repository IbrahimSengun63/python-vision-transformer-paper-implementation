import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()

        self.n_heads = self.config['attention']['n_heads']
        self.head_dimension = self.config['attention']['head_dimension']
        self.embedding_dimension = self.config['model']['embedding_dimension']
        self.attention_drop_rate = self.config['attention']['drop_rate']

        self.attention_dimension = self.calculate_attention_dimension()
        self.projection_layer = self.calculate_projection_layer()
        self.output_layer = self.calculate_output_layer()

    def calculate_attention_dimension(self):
        return self.n_heads * self.head_dimension

    def calculate_projection_layer(self):
        return nn.Linear(self.embedding_dimension, 3 * self.attention_dimension, bias=False)

    def calculate_output_layer(self):
        return nn.Sequential(
            nn.Linear(self.attention_dimension, self.embedding_dimension),
            nn.Dropout(self.attention_drop_rate)
        )

    def load_config(self):
        config_path = 'config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def forward(self, x):
        # Input tensor dimensions: torch.Size([32, 197, 128])
        batch_size, seq_length, emb_dim = x.size()

        # Project inputs into queries, keys, and values
        queries, keys, values = self.projection_layer(x).chunk(3, dim=-1)

        # After projection, each of queries, keys, and values will have shape [32, 197, 384]
        # (assuming self.attention_dimension = 384 after projection)

        # Reshape queries, keys, and values for multi-head attention
        queries = queries.view(batch_size, seq_length, self.n_heads, self.head_dimension).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.n_heads, self.head_dimension).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.n_heads, self.head_dimension).transpose(1, 2)

        # After reshaping and transposing:
        # queries shape: [32, 8, 197, 48] [batch_size, n_heads, seq_length, head_dimension]
        # keys shape: [32, 8, 197, 48] [batch_size, n_heads, seq_length, head_dimension]
        # values shape: [32, 8, 197, 48] [batch_size, n_heads, seq_length, head_dimension]

        # Calculate attention scores
        # queries @ key^T --> [32, 8, 197, 48] @ [32, 8, 48, 197]
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dimension ** 0.5)

        # attention_scores shape: [32, 8, 197, 197] [batch_size, n_heads, seq_length, seq_length]

        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply dropout to attention probabilities (if dropout rate > 0)
        attention_probs = F.dropout(attention_probs, p=self.attention_drop_rate, training=self.training)

        # Weighted sum using attention probabilities
        attention_output = torch.matmul(attention_probs, values)

        # attention_output shape: [32, 8, 197, 48] [batch_size, n_heads, seq_length, head_dimension]

        # Rearrange attention outputs to concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length,
                                                                              self.attention_dimension)

        # After rearranging and reshaping:
        # attention_output shape: [32, 197, 384] [batch_size, seq_length, attention_dimension]
        # (assuming self.attention_dimension = n_heads * head_dimension) (assuming self.attention_dimension = 384)

        # Project attention outputs back to the original embedding dimension
        projected_output = self.output_layer(attention_output)

        # projected_output shape: [32, 197, 128] [batch_size, seq_length, embedding_dimension]
        # (assuming self.embedding_dimension = 128)

        return projected_output
