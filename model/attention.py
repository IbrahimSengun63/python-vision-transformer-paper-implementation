import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.verbose = self.config['train']['verbose']

        self.n_heads = self.config['attention']['n_heads']
        self.head_dimension = self.config['attention']['head_dimension']
        self.embedding_dimension = self.config['model']['embedding_dimension']
        self.attention_drop_rate = self.config['attention']['drop_rate']

        self.attention_dimension = self.calculate_attention_dimension()
        self.projection_layer = self.calculate_projection_layer()
        self.output_layer = self.calculate_output_layer()

    def calculate_attention_dimension(self):
        attention_dim = self.n_heads * self.head_dimension
        if self.verbose:
            print(f"Attention dimension calculated: {attention_dim}")
        return attention_dim

    def calculate_projection_layer(self):
        if self.verbose:
            print("Calculating projection layer...")
        layer = nn.Linear(self.embedding_dimension, 3 * self.attention_dimension, bias=False)
        if self.verbose:
            print(f"Projection layer output shape: {layer(torch.zeros(1, self.embedding_dimension)).shape}")
        return layer

    def calculate_output_layer(self):
        if self.verbose:
            print("Calculating output layer...")
        layer = nn.Sequential(
            nn.Linear(self.attention_dimension, self.embedding_dimension),
            nn.Dropout(self.attention_drop_rate)
        )
        if self.verbose:
            print("Output layer created.")
        return layer

    def load_config(self):
        config_path = 'config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def forward(self, x):
        if self.verbose:
            print(f"Input tensor dimensions: {x.size()}")  # Example: torch.Size([32, 197, 128])
        batch_size, seq_length, emb_dim = x.size()

        # Project inputs into queries, keys, and values
        projection = self.projection_layer(x)
        if self.verbose:
            print(f"After projection, shape: {projection.size()}")

        queries, keys, values = projection.chunk(3, dim=-1)
        if self.verbose:
            print(f"Queries shape: {queries.size()}, Keys shape: {keys.size()}, Values shape: {values.size()}")

        # Reshape queries, keys, and values for multi-head attention
        queries = queries.view(batch_size, seq_length, self.n_heads, self.head_dimension).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.n_heads, self.head_dimension).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.n_heads, self.head_dimension).transpose(1, 2)
        if self.verbose:
            print(f"After reshaping and transposing - Queries shape: {queries.size()}, Keys shape: {keys.size()}, "
                  f"Values shape: {values.size()}")

        # Calculate attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dimension ** 0.5)
        if self.verbose:
            print(f"Attention scores shape: {attention_scores.size()}")

        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        if self.verbose:
            print(f"After softmax - Attention probabilities shape: {attention_probs.size()}")

        # Apply dropout to attention probabilities (if dropout rate > 0)
        attention_probs = F.dropout(attention_probs, p=self.attention_drop_rate, training=self.training)
        if self.verbose:
            print(f"After dropout - Attention probabilities shape: {attention_probs.size()}")

        # Weighted sum using attention probabilities
        attention_output = torch.matmul(attention_probs, values)
        if self.verbose:
            print(f"After weighted sum - Attention output shape: {attention_output.size()}")

        # Rearrange attention outputs to concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length,
                                                                              self.attention_dimension)
        if self.verbose:
            print(f"After rearranging and reshaping - Attention output shape: {attention_output.size()}")

        # Project attention outputs back to the original embedding dimension
        projected_output = self.output_layer(attention_output)
        if self.verbose:
            print(f"After output layer - Projected output shape: {projected_output.size()}")

        return projected_output
