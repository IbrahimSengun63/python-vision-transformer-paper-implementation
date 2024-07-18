import os
import yaml
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset, DatasetDict
from collections import Counter


class CIFAR10CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['img']
        label = torch.tensor(self.dataset[idx]['label']).long()

        if self.transform:
            image = self.transform(image)

        return {'img': image, 'label': label}


class DatasetLoader:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.label_names = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }
        self.train_loader, self.test_loader = self.get_data_loaders()
        self.print_initial_info()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def save_dataset_to_local(self, dataset, local_path):
        os.makedirs(local_path, exist_ok=True)
        dataset.save_to_disk(local_path)

    def load_cifar10_data(self):
        local_path = self.config['dataset']['path']
        dataset_name = self.config['dataset']['name']
        dataset_path = os.path.join(local_path, dataset_name)

        if os.path.isdir(dataset_path):
            dataset = DatasetDict.load_from_disk(dataset_path)
        else:
            dataset = load_dataset(dataset_name)
            if dataset_path:
                self.save_dataset_to_local(dataset, dataset_path)
        return dataset

    def prepare_data(self):
        cifar10_dataset = self.load_cifar10_data()

        transform = transforms.Compose([
            transforms.Resize((self.config['dataset']['img_size'], self.config['dataset']['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_dataset = CIFAR10CustomDataset(cifar10_dataset['train'], transform=transform)
        test_dataset = CIFAR10CustomDataset(cifar10_dataset['test'], transform=transform)

        num_samples_per_class = self.config['loader'].get('num_samples_per_class')
        if num_samples_per_class is not None:
            train_dataset = self.get_balanced_subset(train_dataset, num_samples_per_class)

        return train_dataset, test_dataset

    def get_data_loaders(self):
        train_dataset, test_dataset = self.prepare_data()

        train_loader = DataLoader(train_dataset, batch_size=self.config['loader']['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config['loader']['batch_size'], shuffle=False)

        return train_loader, test_loader

    def get_balanced_subset(self, dataset, num_samples_per_class):
        indices_per_class = {label: [] for label in self.label_names.keys()}

        for idx, sample in enumerate(dataset):
            label = sample['label'].item()
            if len(indices_per_class[label]) < num_samples_per_class:
                indices_per_class[label].append(idx)

        balanced_indices = []
        for indices in indices_per_class.values():
            balanced_indices.extend(indices)

        random.shuffle(balanced_indices)
        return Subset(dataset, balanced_indices)

    def count_label_values(self, dataset):
        label_counts = Counter()
        for sample in dataset:
            label = sample['label'].item()
            label_counts[label] += 1
        return label_counts

    def print_label_distribution(self, dataset):
        label_counts = self.count_label_values(dataset)
        for label, count in label_counts.items():
            print(f"{self.label_names[label]}: {count}")

    def print_initial_info(self):
        print("Dataset Info:")
        print("Label Distribution in Train Loader:")
        self.print_label_distribution(self.train_loader.dataset)
        print("\nBatch Size:", self.train_loader.batch_size)
        print("Batch Shape Example:", next(iter(self.train_loader))['img'].shape)
        print("Length of Train Loader:", len(self.train_loader.dataset))
        print("Length of Test Loader:", len(self.test_loader.dataset))
