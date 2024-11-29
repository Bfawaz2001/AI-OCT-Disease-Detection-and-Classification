import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from src.data.dataset import load_data, OCTDataset
from src.model.model import OCTModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_model(data_dir, model_save_path, num_epochs=10, batch_size=32, learning_rate=0.001):
    logging.info("Loading training and validation data...")

    # Load data
    all_data = load_data(data_dir)
    label_columns = list(all_data.columns[2:-1])  # Select only label columns (excluding ID and image_path)

    # Separate train and validation datasets based on directory structure
    train_data = all_data[all_data["image_path"].str.contains("train")]
    val_data = all_data[all_data["image_path"].str.contains("val")]

    # Create datasets
    train_dataset = OCTDataset(train_data, label_columns=label_columns, transform=None)
    val_dataset = OCTDataset(val_data, label_columns=label_columns, transform=None)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model
    logging.info("Initializing the OCT Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCTModel(num_classes=len(label_columns)).to(device)
    logging.info("Model initialized successfully!")

    # Define loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    logging.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    # Save the model
    logging.info("Training complete. Saving the model...")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    data_dir = "../../data"
    model_save_path = "../../models/oct_model.pth"
    train_model(data_dir, model_save_path, num_epochs=10, batch_size=32, learning_rate=0.001)
