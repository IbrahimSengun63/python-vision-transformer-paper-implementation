import yaml
import torch.nn as nn
from model.patch_embedding import PatchEmbedding
from transformer.transformer_layer import TransformerLayer


class VIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.verbose = self.config['train']['verbose']

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
        if self.verbose:
            print(f"Creating {self.n_layers} transformer layers...")
        # Create a list of transformer layers based on the specified number of layers
        layers = nn.ModuleList([
            TransformerLayer() for _ in range(self.n_layers)
        ])
        if self.verbose:
            print(f"{self.n_layers} transformer layers created.")
        return layers

    def create_dense_layer(self):
        if self.verbose:
            print("Creating dense layer...")
        # Create a sequential dense layer for final classification
        layer = nn.Sequential(
            nn.LayerNorm(self.embedding_dimension),
            nn.Linear(self.embedding_dimension, self.num_classes)
        )
        if self.verbose:
            print("Dense layer created.")
        return layer

    def forward(self, x):
        if self.verbose:
            print(f"Input tensor dimensions: {x.size()}")

        # Forward pass through the model
        x = self.patch_embedding_layer(x)
        if self.verbose:
            print(f"After patch embedding layer, output shape: {x.size()}")

        # Process through each transformer layer in sequence
        for i, layer in enumerate(self.transformer_layers, start=1):
            if self.verbose:
                print(f"Transformer Layer {i}:")
            x = layer(x)

        # Apply final dense layer and return only the first element of each instance's output
        x = self.dense_layer(x)
        if self.verbose:
            print(f"After dense layer, output shape: {x.size()}")
        return x[:, 0]
