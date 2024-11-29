import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

class OCTDataset(Dataset):
    def __init__(self, dataframe, label_columns, transform=None, target_size=(224, 224)):
        self.dataframe = dataframe
        self.label_columns = label_columns
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = str(row['image_path']).strip()

        print(f"Resolved image path: {image_path}")  # Debugging output

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        labels = row[self.label_columns].values.astype(np.float32)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error opening image {image_path}: {e}")

        # Resize the image to target size
        if self.transform:
            image = self.transform(image)
        else:
            image = Compose([Resize(self.target_size), ToTensor()])(image)

        # Normalize image (if needed, add normalization later)
        image = image / 255.0  # Scale pixel values to [0, 1]
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels

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

        print(f"Looking for CSV file: {csv_file}")  # Debugging statement

        if not csv_file.exists():
            print(f"CSV file not found: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        df['image_path'] = df['ID'].apply(lambda x: str(images_dir / f"{x}.png"))
        data.append(df)

    if not data:
        raise FileNotFoundError("No data files were loaded. Check file paths.")

    return pd.concat(data, ignore_index=True)


