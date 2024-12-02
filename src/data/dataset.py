import os

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
import logging
from torchvision.transforms import Compose, Resize, ToTensor, RandomRotation, ColorJitter, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms.functional import gaussian_blur
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to suppress debug logs during normal runs
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class OCTDataset(Dataset):
    def __init__(self, data, label_columns, transform=None, image_dir=None):
        self.data = data
        self.label_columns = label_columns
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row['ID']}.png")
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = np.array(image)  # Ensure it is a NumPy array

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]  # Albumentations expects a NumPy array

        labels = row[self.label_columns].values.astype(float)  # Convert labels to float
        return image, torch.tensor(labels, dtype=torch.float)


def load_data(data_dir):
    subsets = [
        {"name": "train", "csv_name": "RFMiD_Training_Labels.csv"},
        {"name": "val", "csv_name": "RFMiD_Validation_Labels.csv"},
        {"name": "test", "csv_name": "RFMiD_Testing_Labels.csv"},
    ]
    data = []

    for subset in subsets:
        subset_dir = Path(data_dir) / subset["name"]
        csv_file = subset_dir / subset["csv_name"]
        images_dir = subset_dir / "images"

        # Log CSV lookup
        logging.info(f"Looking for CSV file: {csv_file}")

        if not csv_file.exists():
            logging.warning(f"CSV file not found: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        df['image_path'] = df['ID'].apply(lambda x: str(images_dir / f"{x}.png"))
        df['split'] = subset["name"]  # Add a split column to identify the dataset
        data.append(df)

    if not data:
        raise FileNotFoundError("No data files were loaded. Check file paths.")

    return pd.concat(data, ignore_index=True)

