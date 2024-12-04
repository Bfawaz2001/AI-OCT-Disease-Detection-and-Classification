import os
import sys
import torch
from tkinter import filedialog, Label, Button, Tk
from PIL import Image, ImageTk
import torchvision.transforms as transforms

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.model import OCTModel  # Import after fixing the path issue

# Define paths for models
CLASSIFICATION_MODEL_PATH = "../model/disease_classification_model_v15.pth"
DETECTION_MODEL_PATH = "../model/disease_detection_model_v23.pth"

# Load classification model
classification_model = OCTModel(num_classes=7)  # Update num_classes based on your model
classification_model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location="cpu"))
classification_model.eval()

# Load detection model
detection_model = OCTModel(num_classes=1)  # Update num_classes for detection model
detection_model.load_state_dict(torch.load(DETECTION_MODEL_PATH, map_location="cpu"))
detection_model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Prediction function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Detection model
    with torch.no_grad():
        detection_output = torch.sigmoid(detection_model(input_tensor)).item()
        disease_risk = "High Risk" if detection_output > 0.5 else "Low Risk"

    # Classification model
    with torch.no_grad():
        classification_output = torch.sigmoid(classification_model(input_tensor))
        disease_probabilities = classification_output.squeeze().tolist()
        disease_labels = ["DR", "ARMD", "MH", "DN", "MYA", "TSLN", "ODC"]
        disease_results = {disease_labels[i]: prob for i, prob in enumerate(disease_probabilities)}

    return disease_risk, disease_results

# GUI Implementation
def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
    )
    if file_path:
        display_image(file_path)
        disease_risk, disease_results = predict(file_path)
        display_results(disease_risk, disease_results)

def display_image(file_path):
    image = Image.open(file_path).resize((300, 300))
    photo = ImageTk.PhotoImage(image)
    img_label.config(image=photo)
    img_label.image = photo

def display_results(disease_risk, disease_results):
    result_text = f"Disease Risk: {disease_risk}\n"
    result_text += "\nDisease Probabilities:\n"
    for disease, prob in disease_results.items():
        result_text += f"{disease}: {prob:.2f}\n"

    result_label.config(text=result_text)

# Main execution function
def main():
    global img_label, result_label

    # Main window
    window = Tk()
    window.title("OCT AI Disease Detection")
    window.geometry("600x700")

    # GUI Components
    title_label = Label(window, text="OCT AI Disease Detection", font=("Arial", 16))
    title_label.pack(pady=10)

    img_label = Label(window)
    img_label.pack(pady=10)

    upload_button = Button(window, text="Upload Image", command=open_file, font=("Arial", 12))
    upload_button.pack(pady=10)

    result_label = Label(window, text="", font=("Arial", 12), justify="left")
    result_label.pack(pady=10)

    # Run the GUI loop
    window.mainloop()

if __name__ == "__main__":
    main()
