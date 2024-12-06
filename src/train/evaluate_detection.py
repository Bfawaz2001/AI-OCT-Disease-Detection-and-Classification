import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.model.model import OCTModel
from src.data.dataset import OCTDataset
import logging
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO)

def evaluate_model(model_path, test_csv, test_image_dir, label_column, device="cpu"):
    logging.info(f"Using device: {device}")
    logging.info("Loading test data...")

    # Extract model name/version from the file path
    model_name_or_version = os.path.splitext(os.path.basename(model_path))[0]

    # Define the output directory for evaluation graphs
    output_dir = os.path.join("evaluation graphs", "detection", model_name_or_version)
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    test_data = pd.read_csv(test_csv)

    # Data transforms
    test_transform = Compose([
        Resize(224, 224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Create dataset and dataloader
    test_dataset = OCTDataset(
        data=test_data,
        label_columns=[label_column],
        transform=test_transform,
        image_dir=test_image_dir
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    model = OCTModel(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluation
    all_labels = []
    all_predictions = []
    all_probs = []

    logging.info("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device).float().view(-1)
            outputs = torch.sigmoid(model(images)).view(-1)

            # Collect predictions and probabilities
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend((outputs > 0.5).cpu().int().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = (all_predictions == all_labels).mean()
    auc_score = roc_auc_score(all_labels, all_probs)
    classification_report_str = classification_report(all_labels, all_predictions, target_names=["No Disease", "Disease"])

    # Log metrics
    logging.info("Overall Metrics:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"AUC: {auc_score:.4f}")
    logging.info(f"\n{classification_report_str}")

    # Plot and save ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Disease Detection")
    plt.legend(loc="lower right")
    plt.grid()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path)
    logging.info(f"ROC Curve saved to {roc_path}")
    plt.close()

    # Plot and save confusion matrix heatmap
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    logging.info(f"Confusion Matrix saved to {cm_path}")
    plt.close()


if __name__ == "__main__":
    # Paths and parameters
    model_path = "../../model/disease_detection_model_v23.pth"
    test_csv = "../../data/test/RFMiD_Testing_Labels.csv"
    test_image_dir = "../../data/test/images"
    label_column = "Disease_Risk"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate model
    evaluate_model(model_path, test_csv, test_image_dir, label_column, device)
