import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from albumentations import Compose, Resize, Normalize, RandomBrightnessContrast, HorizontalFlip, ShiftScaleRotate, CoarseDropout
from albumentations.pytorch import ToTensorV2
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from src.model.model import OCTModel
from src.data.dataset import OCTDataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)  # Probability for correctly classified
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_logits = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device).float().view(-1)  # Flatten labels
            logits = model(images).view(-1)  # Flatten logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits) > 0.5
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_logits.extend(torch.sigmoid(logits).cpu().numpy())

    total_loss /= len(data_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    auc = roc_auc_score(all_labels, all_logits)

    return total_loss, precision, recall, f1, auc


def train_disease_detection_model(
    train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
    num_epochs, batch_size, learning_rate, device="cpu"
):
    logging.info(f"Using device: {device}")
    logging.info("Loading training and validation data...")

    # Load data
    train_data = pd.read_csv(train_csv)
    val_data = pd.read_csv(val_csv)

    # Class weights for imbalance handling
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=train_data["Disease_Risk"].values
    )
    logging.info(f"Class weights: {class_weights}")

    # Data augmentation
    train_transform = Compose([
        Resize(224, 224),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        RandomBrightnessContrast(p=0.5),
        CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = Compose([
        Resize(224, 224),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Datasets and DataLoaders
    train_dataset = OCTDataset(
        train_data, ["Disease_Risk"], transform=train_transform, image_dir=train_image_dir
    )
    val_dataset = OCTDataset(
        val_data, ["Disease_Risk"], transform=val_transform, image_dir=val_image_dir
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, optimizer, and loss function
    logging.info("Initializing the OCT Model for disease detection...")
    model = OCTModel(num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = FocalLoss(alpha=class_weights[1], gamma=2.0)

    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Training loop
    best_f1 = 0.0
    patience = 5
    best_epoch = 0

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device).float().view(-1)  # Flatten labels
            optimizer.zero_grad()
            logits = model(images).view(-1)  # Flatten logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # Validation
        val_loss, precision, recall, f1, auc = evaluate_model(model, val_loader, criterion, device)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"New best model saved with F1 Score: {f1:.4f}")
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
    model_save_path = "../../model/disease_detection_model_v22.pth"

    # Hyperparameters
    num_epochs = 35
    batch_size = 124
    learning_rate = 0.0001

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train_disease_detection_model(
        train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
        num_epochs, batch_size, learning_rate, device
    )

    model_save_path = "../../model/disease_detection_model_v23.pth"

    # Hyperparameters
    num_epochs = 30
    batch_size = 124
    learning_rate = 0.001

    # Train the model
    train_disease_detection_model(
        train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
        num_epochs, batch_size, learning_rate, device
    )

    model_save_path = "../../model/disease_detection_model_v24.pth"

    # Hyperparameters
    num_epochs = 30
    batch_size = 22
    learning_rate = 0.001


    # Train the model
    train_disease_detection_model(
        train_csv, val_csv, train_image_dir, val_image_dir, model_save_path,
        num_epochs, batch_size, learning_rate, device
    )

