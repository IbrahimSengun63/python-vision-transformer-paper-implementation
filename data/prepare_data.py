import os
import yaml
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
import torch

# Custom dataset class for CIFAR-10
class CIFAR10CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve image and label from dataset
        image = self.dataset[idx]['img']

        label = torch.tensor(self.dataset[idx]['label']).long()

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return {'img': image, 'label': label}


# Class for managing dataset loading and processing
class DatasetLoader:
    def __init__(self):
        # Load configuration from YAML file
        self.config = self.load_config()
        self.train_loader = None
        self.test_loader = None
        # Mapping of label indices to label names
        self.label_names = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }

    # Method to load configuration from YAML file
    def load_config(self):
        config_path = 'config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    # Save dataset to a local directory
    @staticmethod
    def save_dataset_to_local(dataset, local_path):
        os.makedirs(local_path, exist_ok=True)
        dataset.save_to_disk(local_path)

    # Load CIFAR-10 dataset
    def load_cifar10_data(self):
        local_path = self.config['dataset']['path']
        dataset_name = self.config['dataset']['name']
        dataset_path = os.path.join(local_path, dataset_name)

        # Check if dataset is already saved locally, otherwise download
        if dataset_path and os.path.isdir(dataset_path):
            dataset = DatasetDict.load_from_disk(dataset_path)
        else:
            dataset = load_dataset(dataset_name)
            if dataset_path:
                self.save_dataset_to_local(dataset, dataset_path)
        return dataset

    # Prepare CIFAR-10 data for training and testing
    def prepare_data(self):
        cifar10_dataset = self.load_cifar10_data()

        # Define transformations for image preprocessing
        transform = transforms.Compose([
            transforms.Resize((self.config['dataset']['img_size'], self.config['dataset']['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Create custom datasets with transformations
        train_dataset = CIFAR10CustomDataset(cifar10_dataset['train'], transform=transform)
        test_dataset = CIFAR10CustomDataset(cifar10_dataset['test'], transform=transform)

        # Optionally create a balanced subset of training data
        num_samples_per_class = self.config['loader'].get('num_samples_per_class', None)
        if num_samples_per_class is not None and num_samples_per_class != 'None':
            train_dataset = self.get_balanced_subset(train_dataset, num_samples_per_class)

        # Create data loaders for training and testing
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['loader']['batch_size'], shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['loader']['batch_size'], shuffle=False)

        return train_dataset, test_dataset

    # Print lengths of datasets in the data loaders
    def print_dataset_lengths(self):
        if self.train_loader is not None and self.test_loader is not None:
            print(f"Length of train loader: {len(self.train_loader.dataset)}")
            print(f"Length of test loader: {len(self.test_loader.dataset)}")
        else:
            print("Data loaders are not initialized.")

    # Print shapes of batches in the training data loader
    def print_batch_shapes(self):
        if self.train_loader is not None:
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['img']

                batch_size = images.size(0)
                batch_shape = images.shape
                images_shape = images[0].shape

                print(f"Batch {batch_idx + 1}:")
                print(f"Batch size: {batch_size}")
                print(f"Batch shape: {batch_shape}")
                print(f"Shape of the first image tensor: {images_shape}")
                print()
                break
        else:
            print("Train loader is not initialized.")

    # Display a sample image from the training data loader
    def display_image_from_loader(self):
        if self.train_loader is not None:
            for batch in self.train_loader:
                image = batch['img'][0]  # Take the first image from the batch
                label_idx = batch['label'][0].item()  # Take the label corresponding to the first image
                label_name = self.label_names[label_idx]
                image = image.permute(1, 2, 0)  # Rearrange dimensions for matplotlib
                image = (image + 1) / 2  # Un-normalize the image
                plt.imshow(image)
                plt.title(f'Label: {label_name}')
                plt.show()
                break
        else:
            print("Train loader is not initialized.")

    # Get a balanced subset of the dataset with a specified number of samples per class
    def get_balanced_subset(self, dataset, num_samples_per_class):
        class_indices = [[] for _ in range(len(self.label_names))]

        # Collect indices of samples for each class
        for idx, data in enumerate(dataset.dataset):
            label = data['label']
            class_indices[label].append(idx)

        # Create a balanced subset by selecting a fixed number of samples per class
        balanced_indices = []
        for indices in class_indices:
            if len(indices) > num_samples_per_class:
                balanced_indices.extend(indices[:num_samples_per_class])
            else:
                balanced_indices.extend(indices)

        # Create a new dataset with the balanced subset of indices
        balanced_dataset = [dataset[idx] for idx in balanced_indices]
        return balanced_dataset

    # Get counts of each label in the training and test datasets
    def get_label_value_counts(self):
        if self.train_loader is not None:
            label_counts_train = {label_idx: 0 for label_idx in range(len(self.label_names))}
            label_counts_test = {label_idx: 0 for label_idx in range(len(self.label_names))}

            # Count occurrences of each label in the training dataset
            for batch in self.train_loader:
                labels = batch['label']
                for label in labels:
                    label_counts_train[label.item()] += 1

            # Count occurrences of each label in the test dataset
            for batch in self.test_loader:
                labels = batch['label']
                for label in labels:
                    label_counts_test[label.item()] += 1

            # Print label value counts for training and test datasets
            print("Label value counts in training data:")
            for label_idx, count in label_counts_train.items():
                print(f"{self.label_names[label_idx]}: {count}")

            print("\nLabel value counts in test data:")
            for label_idx, count in label_counts_test.items():
                print(f"{self.label_names[label_idx]}: {count}")

            return label_counts_train, label_counts_test
        else:
            print("Train loader is not initialized.")

    # Static method to get an instance of DatasetLoader and return prepared data loaders
    @staticmethod
    def get_loader(verbose=False):
        loader_instance = DatasetLoader()
        train_loader, test_loader = loader_instance.prepare_data()
        if verbose is True:
            loader_instance.print_dataset_lengths()
            loader_instance.print_batch_shapes()
            loader_instance.get_label_value_counts()
        return train_loader, test_loader
