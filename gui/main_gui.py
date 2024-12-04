import os
import sys
import torch
from tkinter import filedialog, Label, Button, Tk, Frame, Toplevel
from tkinter.ttk import Style
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.model import OCTModel  # Import after fixing the path issue

# Define paths for models
CLASSIFICATION_MODEL_PATH = "../model/disease_classification_model_v15.pth"
DETECTION_MODEL_PATH = "../model/disease_detection_model_v23.pth"

# Load classification model
classification_model = OCTModel(num_classes=7)
classification_model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location="cpu"))
classification_model.eval()

# Load detection model
detection_model = OCTModel(num_classes=1)
detection_model.load_state_dict(torch.load(DETECTION_MODEL_PATH, map_location="cpu"))
detection_model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Disease Labels and Full Names
DISEASE_LABELS = {
    "DR": "Diabetic Retinopathy",
    "ARMD": "Age-Related Macular Degeneration",
    "MH": "Macular Hole",
    "DN": "Drusen",
    "MYA": "Myopia",
    "TSLN": "Tessellation",
    "ODC": "Optic Disc Cupping",
}

# Tooltip Class
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 20
        self.tip_window = tw = Tk()
        tw.wm_overrideredirect(True)
        tw.geometry(f"+{x}+{y}")
        label = Label(tw, text=self.text, justify="left",
                      bg="#FFFFE0", fg="black", relief="solid", borderwidth=1,
                      font=("Arial", 12))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

# Prediction function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        # Detection model
        detection_output = torch.sigmoid(detection_model(input_tensor)).item()
        disease_risk = "High Risk" if detection_output > 0.5 else "Low Risk"

        # Classification model
        classification_output = torch.sigmoid(classification_model(input_tensor))
        disease_probabilities = classification_output.squeeze().tolist()
        disease_results = {label: prob for label, prob in zip(DISEASE_LABELS.keys(), disease_probabilities)}

    return disease_risk, disease_results

# GUI Functions
def open_file():
    global current_image_path
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
    )
    if file_path:
        current_image_path = file_path
        display_image(file_path)
        disease_risk, disease_results = predict(file_path)
        display_results(disease_risk, disease_results)
        show_graph(file_path, disease_results)

def display_image(file_path):
    image = Image.open(file_path)
    max_size = 400
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    img_label.config(image=photo, text="")
    img_label.image = photo
    img_size_label.config(text=f"Original Image Size: {image.size}")

def display_results(disease_risk, disease_results):
    risk_color = "green" if disease_risk == "Low Risk" else "red"
    result_text = f"Disease Risk: {disease_risk}\n\nDisease Probabilities:\n"
    for disease, prob in disease_results.items():
        result_text += f"{disease}: {prob:.2f}\n"
    result_label.config(text=result_text, fg=risk_color)

    # Add summary insights
    summary_text = "Summary Insights:\n"
    high_prob_diseases = [label for label, prob in disease_results.items() if prob > 0.5]

    if disease_risk == "Low Risk" and high_prob_diseases:
        summary_text += "Low Risk detected but probabilities suggest possible diseases:\n"
        for label in high_prob_diseases:
            summary_text += f"{label} ({DISEASE_LABELS[label]}): High Probability\n"
        summary_text += "Recommended additional screening, and professional opinion.\n"

    elif disease_risk == "High Risk" and not high_prob_diseases:
        summary_text += ("High Risk detected but no significant disease probabilities for tested diseases found. "
                         "Recommended additional screening, and professional opinion.\n")

    elif high_prob_diseases:
        summary_text += "High Probability Diseases:\n"
        for label in high_prob_diseases:
            summary_text += f"{label} ({DISEASE_LABELS[label]}): High Probability\n"

    else:
        summary_text += "No significant matches found.\n"

    summary_label.config(text=summary_text)

def show_graph(file_path, disease_results):
    try:
        if isinstance(disease_results, dict):
            labels = list(disease_results.keys())
            probabilities = list(disease_results.values())
            file_name = os.path.basename(file_path)
            graph_file_name = f"{os.path.splitext(file_name)[0]}_disease_probabilities_graph.png"

            # Create a new window for the graph
            graph_window = Toplevel()
            graph_window.title("Disease Probabilities Chart")
            graph_window.geometry("1000x700")
            graph_window.configure(bg="#1E1E1E")

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(labels, probabilities, color='skyblue', alpha=0.9, edgecolor="black")
            ax.set_xlabel("Diseases", fontsize=14)
            ax.set_ylabel("Probability", fontsize=14)
            ax.set_title("Disease Probabilities", fontsize=16, fontweight="bold")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # Add probabilities on top of the bars
            for bar, prob in zip(bars, probabilities):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{prob:.2f}", ha="center", fontsize=12)

            canvas = FigureCanvasTkAgg(fig, master=graph_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack()

            # Save Button
            def save_graph():
                fig.savefig(graph_file_name)
                save_label.config(text=f"Graph saved as {graph_file_name}", fg="green")

            save_button = Button(
                graph_window, text="Save Graph", command=save_graph, font=("Arial", 12),
                cursor="hand2"
            )
            save_button.pack(pady=10)

            save_label = Label(
                graph_window, text="", font=("Arial", 12), bg="#1E1E1E", fg="white"
            )
            save_label.pack()

    except Exception as e:
        print(f"Error displaying graph: {e}")  # Debug
        result_label.config(text=f"Error displaying graph: {e}")

def show_help():
    help_window = Toplevel()
    help_window.title("Help")
    help_window.geometry("400x400")
    help_window.configure(bg="#1E1E1E")

    help_label = Label(
        help_window, text="Help", font=("Arial", 16, "bold"), bg="#1E1E1E", fg="white"
    )
    help_label.pack(pady=10)

    help_text = Label(
        help_window,
        text="Instructions:\n\n"
             "1. Click 'Upload Image' to select an OCT image from your computer.\n\n"
             "2. The app displays:\n"
             "   - Disease risk (High Risk or Low Risk).\n"
             "   - Probabilities for specific diseases.\n\n"
             "3. A graph of disease probabilities is shown in a separate window.\n\n"
             "4. You can save the graph as an image file.\n\n"
             "Disclaimer:\n"
             "This tool is for research purposes only and should not be used for diagnosing patients."
             "Consult a medical professional for clinical advice.",
        font=("Arial", 14), bg="#1E1E1E", fg="white", wraplength=350, justify="left"
    )
    help_text.pack(pady=10)

# Main Execution Function
def main():
    global img_label, result_label, img_size_label, summary_label, current_image_path

    # Main window
    window = Tk()
    window.title("OCT Disease Detection and Classification")
    window.geometry("800x1000")
    window.configure(bg="#1E1E1E")

    # Help Button
    help_button = Button(
        window, text="Help", command=show_help, font=("Arial", 12), cursor="hand2"
    )
    help_button.place(relx=1, x=-10, y=10, anchor="ne")

    # Title
    title_label = Label(
        window, text="OCT Disease Detection and Classification",
        font=("Helvetica", 20, "bold"), fg="white", bg="#1E1E1E", pady=10
    )
    title_label.pack()

    # Image Display Section
    img_frame = Frame(window, bg="#1E1E1E")
    img_frame.pack(pady=10)
    img_label = Label(img_frame, text="Upload an OCT Image", font=("Arial", 16, "italic"), bg="#1E1E1E", fg="white")
    img_label.pack()
    img_size_label = Label(img_frame, text="", font=("Arial", 14), bg="#1E1E1E", fg="gray")
    img_size_label.pack()

    # Upload Button
    upload_button = Button(
        window, text="Upload Image", command=open_file, font=("Arial", 14),  cursor="hand2"
    )
    upload_button.pack(pady=10)
    ToolTip(upload_button, "Click to upload an OCT image.")

    # Results Section
    results_frame = Frame(window, bg="#1E1E1E")
    results_frame.pack(fill="both", expand=True, pady=10)
    result_label = Label(
        results_frame, text="", font=("Arial", 14), justify="left",
        bg="#1E1E1E", fg="white", wraplength=700
    )
    result_label.pack()

    # Summary Section
    summary_frame = Frame(window, bg="#1E1E1E")
    summary_frame.pack(fill="both", expand=True, pady=10)
    summary_label = Label(
        summary_frame, text="", font=("Arial", 14), justify="left",
        bg="#1E1E1E", fg="white", wraplength=700
    )
    summary_label.pack()

    window.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error in application: {e}")
