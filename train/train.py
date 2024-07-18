import os
import torch
import torch.nn as nn
import yaml
import torch.optim as optim


class VisionTransformer:
    def __init__(self, model, train_loader, test_loader, device=None, model_name="vit_model"):
        self.config = self.load_config()
        self.epoch = self.config['train']['epoch']
        self.lr = self.config['train']['lr']

        self.model = model
        self.model_name = model_name

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        # Create directory for saving models if it does not exist
        self.save_dir = 'trained_models/' + self.model_name
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def load_config(self):
        config_path = 'config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def train_for_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['img'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        train_loss /= len(self.train_loader)
        return train_loss

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                images = batch['img'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(self.test_loader)
        accuracy = 100 * correct / total
        return val_loss, accuracy

    def save_model(self, epoch, lr):
        model_filename = f"{self.model_name}_epoch{epoch}_lr{lr}.pth"
        save_path = os.path.join(self.save_dir, model_filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded model from {model_path}")

    def run_training(self):
        for epoch in range(1, self.epoch + 1):
            train_loss = self.train_for_one_epoch()
            val_loss, val_accuracy = self.validate()

            print(
                f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")

            # Save model after each epoch
            self.save_model(epoch, self.optimizer.param_groups[0]['lr'])
