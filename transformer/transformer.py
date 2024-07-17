import yaml
import torch.nn as nn
from model.patch_embedding import PatchEmbedding
from transformer.transformer_layer import TransformerLayer


class VIT(nn.Module):
    def __init__(self):
        super().__init__()

        # Load configuration from YAML file
        self.config = self.load_config()

        # Extract relevant parameters from the configuration
        self.n_layers = self.config['transformer']['n_layers']
        self.embedding_dimension = self.config['model']['embedding_dimension']
        self.num_classes = self.config['dataset']['num_classes']

        # Initialize patch embedding layer
        self.patch_embedding_layer = PatchEmbedding()

        # Create transformer layers based on the specified number of layers
        self.transformer_layers = self.create_transformer_layer()

        # Create a dense layer for final classification
        self.dense_layer = self.create_dense_layer()

    def load_config(self):
        # Load configuration file
        config_path = 'config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def create_transformer_layer(self):
        # Create a list of transformer layers based on the specified number of layers
        return nn.ModuleList([
            TransformerLayer() for _ in range(self.n_layers)
        ])

    def create_dense_layer(self):
        # Create a sequential dense layer for final classification
        return nn.Sequential(
            nn.LayerNorm(self.embedding_dimension),
            nn.Linear(self.embedding_dimension, self.num_classes)
        )

    def forward(self, x):
        # Forward pass through the model
        x = self.patch_embedding_layer(x)

        # Process through each transformer layer in sequence
        for layer in self.transformer_layers:
            x = layer(x)

        # Apply final dense layer and return only the first element of each instance's output
        x = self.dense_layer(x)
        return x[:, 0]
