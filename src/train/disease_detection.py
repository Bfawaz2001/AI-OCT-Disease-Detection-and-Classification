import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from src.model.model import OCTModel
from src.data.dataset import OCTDataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def train_disease_detection_model(
    train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
    num_epochs, batch_size, learning_rate, device="cpu"
):
    logging.info(f"Using device: {device}")
    logging.info("Loading training and validation data...")

    # Load data
    train_data = pd.read_csv(train_csv)
    val_data = pd.read_csv(val_csv)

    # Determine class weights to handle class imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),  # Convert to NumPy array
        y=train_data["Disease_Risk"].values
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights)}
    logging.info(f"Class weights: {class_weights}")

    # Data transforms
    train_transform = Compose([
        Resize(224, 224),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = Compose([
        Resize(224, 224),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Create datasets
    train_dataset = OCTDataset(
        train_data, ["Disease_Risk"], transform=train_transform, image_dir=train_image_dir
    )
    val_dataset = OCTDataset(
        val_data, ["Disease_Risk"], transform=val_transform, image_dir=val_image_dir
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    logging.info("Initializing the OCT Model for disease detection...")
    model = OCTModel(num_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights[1], device=device))

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, verbose=True
    )

    # Training loop
    best_val_loss = float("inf")
    patience = 5
    best_epoch = 0

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            images = images.to(device)
            labels = labels.to(device).float().view(-1)  # Flatten labels to match output size

            optimizer.zero_grad()
            outputs = model(images).view(-1)  # Flatten outputs to match label size
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}", unit="batch"):
                images = images.to(device)
                labels = labels.to(device).float().view(-1)  # Flatten labels
                outputs = model(images).view(-1)  # Flatten outputs
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # Adjust learning rate
        scheduler.step(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"New best model saved with Validation Loss: {val_loss:.4f}")
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")
            break


if __name__ == "__main__":
    # Paths
    train_csv = "../../data/detection/RFMiD_Training_Detection.csv"
    val_csv = "../../data/detection/RFMiD_Validation_Detection.csv"
    train_image_dir = "../../data/train/images"
    val_image_dir = "../../data/val/images"
    model_save_path = "../../model/disease_detection_model_v17.pth"

    # Hyperparameters
    num_epochs = 40
    batch_size = 20
    learning_rate = 0.001

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train_disease_detection_model(
        train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
        num_epochs, batch_size, learning_rate, device
    )
