import torch
import torch.nn as nn
import yaml


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.config = self.load_config()
        self.verbose = self.config['train']['verbose']

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
        num_patches = (self.img_height // self.patch_height) * (self.img_width // self.patch_width)
        if self.verbose:
            print(f"Number of patches calculated: {num_patches}")
        return num_patches

    def calculate_patch_dimension(self):
        patch_dim = self.img_channels * self.patch_height * self.patch_width
        if self.verbose:
            print(f"Patch dimension calculated: {patch_dim}")
        return patch_dim

    def create_patch_embedding_layer(self):
        if self.verbose:
            print("Creating patch embedding layer...")
        layer = nn.Sequential(
            nn.LayerNorm(self.patch_dimension),
            nn.Linear(self.patch_dimension, self.embedding_dimension),
            nn.LayerNorm(self.embedding_dimension)
        )
        if self.verbose:
            print(f"Patch embedding layer created. Output shape: {layer(torch.zeros(1, self.patch_dimension)).shape}")
        return layer

    def create_positional_embedding(self):
        pos_embedding = nn.Parameter(torch.zeros(1, self.number_of_patches + 1, self.embedding_dimension))
        cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dimension))
        patch_embedding_dropout = nn.Dropout(self.patch_embedding_drop_rate)
        if self.verbose:
            print(f"Positional embedding created. Pos embedding shape: {pos_embedding.shape}, "
                  f"Cls token shape: {cls_token.shape}")
        return pos_embedding, cls_token, patch_embedding_dropout

    def forward(self, x):
        batch_size = x.shape[0]  # Get the batch size
        if self.verbose:
            print(f"Input shape: {x.shape}")
        x = self.create_patches(x, batch_size)
        if self.verbose:
            print(f"Patches created. Output shape: {x.shape}")
        x = self.patch_embedding_layer(x)
        if self.verbose:
            print(f"After patch embedding layer. Output shape: {x.shape}")
        x = self.add_class_token(x, batch_size)
        if self.verbose:
            print(f"After adding class token. Output shape: {x.shape}")
        x = self.add_positional_embedding(x)
        if self.verbose:
            print(f"After adding positional embedding. Output shape: {x.shape}")
        x = self.patch_embedding_dropout(x)
        if self.verbose:
            print(f"After dropout. Output shape: {x.shape}")
        return x

    def create_patches(self, x, batch_size):
        x = x.reshape(batch_size, self.img_channels, self.img_height, self.img_width)
        if self.verbose:
            print(f"Reshaped input to patches. Shape: {x.shape}")
        x = x.unfold(2, self.patch_height, self.patch_height).unfold(3, self.patch_width, self.patch_width)
        if self.verbose:
            print(f"Unfolded patches. Shape: {x.shape}")
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, self.patch_dimension)
        if self.verbose:
            print(f"Permuted patches. Shape: {x.shape}")
        return x

    def add_class_token(self, x, batch_size):
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.verbose:
            print(f"Added class token. Output shape: {x.shape}")
        return x

    def add_positional_embedding(self, x):
        x = x + self.pos_embedding
        if self.verbose:
            print(f"Added positional embedding. Output shape: {x.shape}")
        return x
