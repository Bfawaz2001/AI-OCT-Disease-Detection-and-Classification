import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from albumentations import Compose, Resize, Normalize, RandomRotate90, Flip, RandomBrightnessContrast, ElasticTransform, \
    CoarseDropout, HorizontalFlip
from albumentations.pytorch import ToTensorV2
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from src.model.model import OCTModel
from src.data.dataset import OCTDataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def train_classification_model(
    train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
    num_epochs, batch_size, learning_rate, device="cpu"
):
    logging.info(f"Using device: {device}")
    logging.info("Loading training and validation data...")

    # Load data
    train_data = pd.read_csv(train_csv)
    val_data = pd.read_csv(val_csv)

    # Compute class weights
    class_weights = torch.tensor([
        compute_class_weight("balanced", classes=np.array([0, 1]), y=train_data[label].values)[1]
        for label in train_data.columns if label != "ID"
    ], dtype=torch.float).to(device)
    logging.info(f"Class weights: {class_weights}")

    # Data augmentation
    train_transform = Compose([
        Resize(224, 224),
        RandomRotate90(p=0.5),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5),
        ElasticTransform(alpha=1.0, sigma=50, p=0.5),  # Removed invalid alpha_affine
        CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = Compose([
        Resize(224, 224),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Create datasets and data loaders
    label_columns = [col for col in train_data.columns if col != "ID"]
    train_dataset = OCTDataset(train_data, label_columns, transform=train_transform, image_dir=train_image_dir)
    val_dataset = OCTDataset(val_data, label_columns, transform=val_transform, image_dir=val_image_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    logging.info("Initializing the OCT Model for classification...")
    model = OCTModel(num_classes=len(label_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Training loop
    best_val_f1 = 0.0
    patience = 5
    best_epoch = 0

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}", unit="batch"):
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_f1 = f1_score(np.array(all_labels), np.array(all_preds), average="weighted")
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, F1 Score: {val_f1:.4f}")

        # Save the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"New best model saved with F1 Score: {val_f1:.4f}")
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")
            break


if __name__ == "__main__":
    # Paths
    train_csv = "../../data/classification/RFMiD_Training_Classification.csv"
    val_csv = "../../data/classification/RFMiD_Validation_Classification.csv"
    train_image_dir = "../../data/train/images"
    val_image_dir = "../../data/val/images"
    model_save_path = "../../model/disease_classification_model_v17.pth"

    # Hyperparameters
    num_epochs = 30
    batch_size = 124
    learning_rate = 0.0001

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train_classification_model(
        train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
        num_epochs, batch_size, learning_rate, device
    )

    model_save_path = "../../model/disease_classification_model_v18.pth"

    # Hyperparameters
    num_epochs = 30
    batch_size = 124
    learning_rate = 0.001

    # Train the model
    train_classification_model(
        train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
        num_epochs, batch_size, learning_rate, device
    )

    model_save_path = "../../model/disease_classification_model_v19.pth"

    # Hyperparameters
    num_epochs = 30
    batch_size = 22
    learning_rate = 0.005

    # Train the model
    train_classification_model(
        train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
        num_epochs, batch_size, learning_rate, device
    )

    model_save_path = "../../model/disease_classification_model_v20.pth"

    # Hyperparameters
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.001

    # Train the model
    train_classification_model(
        train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
        num_epochs, batch_size, learning_rate, device
    )

    model_save_path = "../../model/disease_classification_model_v21.pth"

    # Hyperparameters
    num_epochs = 30
    batch_size = 64
    learning_rate = 0.0001

    # Train the model
    train_classification_model(
        train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
        num_epochs, batch_size, learning_rate, device
    )

    model_save_path = "../../model/disease_classification_model_v22.pth"

    # Hyperparameters
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.0001

    # Train the model
    train_classification_model(
        train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
        num_epochs, batch_size, learning_rate, device
    )