import torch.nn as nn
from model.attention import Attention
import yaml


class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()

        # Initialize dimensions from configuration
        self.embedding_dimension = self.config['model']['embedding_dimension']
        self.ffn_hidden_dimension = self.config['transformer']['ffn_hidden_dimension']
        self.ffn_drop_rate = self.config['transformer']['drop_rate']

        # Create attention and FFN blocks
        self.attention_block = self.create_attention_block()
        self.ffn_block = self.create_ffn_block()

    def load_config(self):
        # Load configuration file
        config_path = 'config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def create_attention_block(self):
        # Create attention block: layer normalization -> attention mechanism -> layer normalization
        return nn.Sequential(
            nn.LayerNorm(self.embedding_dimension),
            Attention(),
            nn.LayerNorm(self.embedding_dimension)
        )

    def create_ffn_block(self):
        # Create feed-forward neural network block: linear -> ReLU -> dropout -> linear -> dropout
        return nn.Sequential(
            nn.Linear(self.embedding_dimension, self.ffn_hidden_dimension),
            nn.ReLU(),
            nn.Dropout(self.ffn_drop_rate),
            nn.Linear(self.ffn_hidden_dimension, self.embedding_dimension),
            nn.Dropout(self.ffn_drop_rate)
        )

    def forward(self, x):
        # Forward pass through the transformer layer

        # Apply attention block
        x_attention = self.attention_block(x)

        # Add residual connection and layer normalization
        x = x + x_attention

        # Apply FFN block
        x_ffn = self.ffn_block(x)

        # Add residual connection and layer normalization
        x = x + x_ffn

        return x
