import torch
import numpy as np
from src.data.preprocess import preprocess_image
from src.model.model import OCTModel


def predict_image(image_path, model_path, device='cuda'):
    """
    Predict diseases from a single OCT image.

    Args:
        image_path (str): Path to the OCT image.
        model_path (str): Path to the trained model file.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        dict: Predicted probabilities for each disease.
    """
    # Define disease labels
    disease_labels = [
        "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
        "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
        "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"
    ]

    # Load and preprocess the image
    print(f"Loading image: {image_path}")
    image = preprocess_image(image_path)  # Defined in preprocess.py
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
    image = image.to(device)

    # Load the model
    print("Loading model...")
    model = OCTModel(num_classes=len(disease_labels))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Make predictions
    print("Making predictions...")
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]  # Convert logits to probabilities

    # Map predictions to disease labels
    predictions = {label: prob for label, prob in zip(disease_labels, probabilities)}

    return predictions


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict diseases from a single OCT image.")
    parser.add_argument("image_path", type=str, help="Path to the OCT image")
    parser.add_argument("model_path", type=str, help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (default: 'cuda' if available)")

    args = parser.parse_args()

    # Run prediction
    predictions = predict_image(args.image_path, args.model_path, args.device)
    print("\nPredictions:")
    for disease, prob in predictions.items():
        print(f"{disease}: {prob:.4f}")
