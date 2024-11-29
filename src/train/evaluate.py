import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data.dataset import OCTDataset, load_data
from src.model.model import OCTModel


def load_test_data():
    """
    Load the test dataset from a hardcoded path.

    Returns:
        DataLoader: PyTorch DataLoader for the test dataset.
    """
    data_dir = "../../data/test"
    from pathlib import Path
    from src.data.dataset import OCTDataset

    csv_file = Path(data_dir) / "RFMiD_Testing_Labels.csv"
    images_dir = Path(data_dir) / "images"

    print(f"Looking for CSV file: {csv_file}")
    if not csv_file.exists():
        raise FileNotFoundError(f"Test CSV file not found at {csv_file}")

    # Load CSV and link image paths
    import pandas as pd
    df = pd.read_csv(csv_file)
    df['image_path'] = df['ID'].apply(lambda x: str(images_dir / f"{x}.png"))

    # Create dataset and dataloader
    test_dataset = OCTDataset(df)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_dataloader


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the test dataset.
        device (str): Device to run the evaluation on.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).float()  # Binarize outputs at 0.5

            # Store predictions and labels
            all_labels.append(labels.cpu())
            all_preds.append(predictions.cpu())

    # Convert to numpy arrays
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}


if __name__ == "__main__":
    data_dir = "../../data/test"
    model_path = "../../model/oct_model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the test data
    print("Loading test data...")
    test_dataloader = load_test_data()

    # Load the model
    print("Loading model...")
    model = OCTModel(num_classes=28)  # Adjust based on the number of disease classes
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_dataloader, device)

    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
