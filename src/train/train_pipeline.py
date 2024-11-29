import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.data.dataset import OCTDataset
from src.models.model import MyModel  # Replace with your model
from src.data.utils import compute_class_weights
import torch.nn as nn

def train_model(train_loader, val_loader, model, criterion, optimizer, device):
    num_epochs = 10
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader)}")
