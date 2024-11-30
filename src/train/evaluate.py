import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.model.model import OCTModel  # Adjust the import based on your project structure
from src.data.dataset import OCTDataset, load_data  # Adjust the import based on your project structure
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

# Define evaluation function
def evaluate_model(model_path, test_loader, label_columns, device):
    # Load the model with matching output dimensions
    model = OCTModel(num_classes=len(label_columns)).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model_state_dict = checkpoint

    # Adjust the state dict for size mismatches
    model_state_dict["base_model.fc.3.weight"] = model_state_dict["base_model.fc.3.weight"][:len(label_columns)]
    model_state_dict["base_model.fc.3.bias"] = model_state_dict["base_model.fc.3.bias"][:len(label_columns)]

    model.load_state_dict(model_state_dict, strict=False)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # Threshold for multi-label classification

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute metrics
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_columns))

    # Confusion Matrix (for single-label classification or dominant label in multi-label)
    cm = confusion_matrix(all_labels.argmax(axis=1), all_preds.argmax(axis=1))
    plot_confusion_matrix(cm, label_columns)

    # Compute ROC-AUC if applicable
    if len(label_columns) > 1:
        roc_auc = roc_auc_score(all_labels, all_preds, average='macro')
        print(f"ROC-AUC Score: {roc_auc:.4f}")

# Plot confusion matrix
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Configurations
    model_path = "../../model/oct_model.pth"  # Path to your best saved model
    data_dir = "../../data"  # Adjust to your data directory
    label_columns = ['DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS', 'MS',
                     'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST', 'AION', 'PT',
                     'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'OTHER']  # Adjust this list as needed
    batch_size = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    transform = Compose([Resize((224, 224)), ToTensor()])
    data = load_data(data_dir)
    test_data = data[data["split"] == "test"]
    test_dataset = OCTDataset(test_data, label_columns, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate model
    evaluate_model(model_path, test_loader, label_columns, device)
