import torch
import torch.nn as nn
import yaml


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()

        # Load dataset and model parameters from the config file
        self.img_height = self.config['dataset']['img_size']
        self.img_width = self.config['dataset']['img_size']
        self.img_channels = self.config['dataset']['img_channels']
        self.embedding_dimension = self.config['model']['embedding_dimension']
        self.patch_embedding_drop_rate = self.config['model']['patch_embedding_drop_rate']
        self.patch_height = self.config['model']['patch_height']
        self.patch_width = self.config['model']['patch_width']

        # Calculate number of patches and patch dimension
        self.number_of_patches = self.calculate_number_of_patches()
        self.patch_dimension = self.calculate_patch_dimension()

        # Create layers and embeddings
        self.patch_embedding_layer = self.create_patch_embedding_layer()
        self.pos_embedding, self.cls_token, self.patch_embedding_dropout = self.create_positional_embedding()

    def load_config(self):
        config_path = 'config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def calculate_number_of_patches(self):
        # Calculate the number of patches from image dimensions and patch size
        return (self.img_height // self.patch_height) * (self.img_width // self.patch_width)

    def calculate_patch_dimension(self):
        # Calculate the dimension of each patch
        return self.img_channels * self.patch_height * self.patch_width

    def create_patch_embedding_layer(self):
        # Create a layer to embed patches
        return nn.Sequential(
            nn.LayerNorm(self.patch_dimension),
            nn.Linear(self.patch_dimension, self.embedding_dimension),
            nn.LayerNorm(self.embedding_dimension)
        )

    def create_positional_embedding(self):
        # Create positional embedding, class token, and dropout layer
        pos_embedding = nn.Parameter(torch.zeros(1, self.number_of_patches + 1, self.embedding_dimension))
        cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dimension))
        patch_embedding_dropout = nn.Dropout(self.patch_embedding_drop_rate)
        return pos_embedding, cls_token, patch_embedding_dropout

    def forward(self, x):
        batch_size = x.shape[0]  # Get the batch size
        x = self.create_patches(x, batch_size)  # Create patches from the input
        x = self.patch_embedding_layer(x)  # Apply patch embedding layer
        x = self.add_class_token(x, batch_size)  # Add class token
        x = self.add_positional_embedding(x)  # Add positional embedding
        x = self.patch_embedding_dropout(x)  # Apply dropout
        return x

    def create_patches(self, x, batch_size):
        # Reshape and permute the input to create patches
        # (32, 3, 224, 224) (batch_size, num_channels, img_height, img_width)
        x = x.reshape(batch_size, self.img_channels, self.img_height, self.img_width)

        # (32, 3, 14, 14, 16, 16) (batch_size, num_channels, num_patches_height, num_patches_width, patch_height, patch_width)
        x = x.unfold(2, self.patch_height, self.patch_height).unfold(3, self.patch_width, self.patch_width)

        # (32, 196, 768) (batch_size, num_patches, patch_dimension)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, self.patch_dimension)
        return x

    def add_class_token(self, x, batch_size):
        # Add a class token to the sequence of patches
        # (32, 1, 768) (batch_size, 1, embedding_dimension)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # (32, 197, 768) (batch_size, num_patches + 1, patch_dimension)
        return torch.cat((cls_tokens, x), dim=1)

    def add_positional_embedding(self, x):
        # Add positional embedding to the sequence
        # (32, 197, 768) (batch_size, num_patches + 1, patch_dimension)
        return x + self.pos_embedding
