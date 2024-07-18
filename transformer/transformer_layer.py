import torch.nn as nn
from model.attention import Attention
import yaml


class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.verbose = self.config['train']['verbose']

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
        if self.verbose:
            print("Creating attention block...")
        # Create attention block: layer normalization -> attention mechanism -> layer normalization
        block = nn.Sequential(
            nn.LayerNorm(self.embedding_dimension),
            Attention(),  # Pass verbose flag to Attention module
            nn.LayerNorm(self.embedding_dimension)
        )
        if self.verbose:
            print("Attention block created.")
        return block

    def create_ffn_block(self):
        if self.verbose:
            print("Creating FFN block...")
        # Create feed-forward neural network block: linear -> ReLU -> dropout -> linear -> dropout
        block = nn.Sequential(
            nn.Linear(self.embedding_dimension, self.ffn_hidden_dimension),
            nn.ReLU(),
            nn.Dropout(self.ffn_drop_rate),
            nn.Linear(self.ffn_hidden_dimension, self.embedding_dimension),
            nn.Dropout(self.ffn_drop_rate)
        )
        if self.verbose:
            print("FFN block created.")
        return block

    def forward(self, x):
        if self.verbose:
            print(f"Input tensor dimensions: {x.size()}")

        # Apply attention block
        x_attention = self.attention_block(x)
        if self.verbose:
            print(f"After attention block, output shape: {x_attention.size()}")

        # Add residual connection and layer normalization
        x = x + x_attention
        if self.verbose:
            print(f"After adding residual connection and layer normalization, output shape: {x.size()}")

        # Apply FFN block
        x_ffn = self.ffn_block(x)
        if self.verbose:
            print(f"After FFN block, output shape: {x_ffn.size()}")

        # Add residual connection and layer normalization
        x = x + x_ffn
        if self.verbose:
            print(f"After adding residual connection and layer normalization, output shape: {x.size()}")

        return x
