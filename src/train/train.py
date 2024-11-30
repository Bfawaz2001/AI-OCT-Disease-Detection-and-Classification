import os
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from src.model.model import OCTModel
from src.data.dataset import OCTDataset, load_data
from torchvision.transforms import Compose, Resize, ToTensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_model(data_dir, model_save_path, num_epochs=10, batch_size=32, learning_rate=0.001):
    logging.info("Loading training and validation data...")

    # Load the full dataset
    data = load_data(data_dir)
    label_columns = [col for col in data.columns if
                     col not in ['ID', 'image_path', 'split']]  # Exclude non-label columns

    # Split the dataset into train and validation
    train_data = data[data["split"] == "train"]
    val_data = data[data["split"] == "val"]

    # Create datasets and dataloaders
    train_dataset = OCTDataset(train_data, label_columns, transform=None)
    val_dataset = OCTDataset(val_data, label_columns, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logging.info("Initializing the OCT Model...")
    model = OCTModel(num_classes=len(label_columns))
    logging.info("Model initialized successfully!")

    # Loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logging.info("Starting training...")
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"New best model saved with Validation Loss: {best_loss:.4f}")


if __name__ == "__main__":
    data_dir = "../../data"  # Update this to your actual data directory
    model_save_path = "../../model/oct_model.pth"  # Path to save the final model
    train_model(data_dir, model_save_path, num_epochs=50, batch_size=10, learning_rate=0.001)
