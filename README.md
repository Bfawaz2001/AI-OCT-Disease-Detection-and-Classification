
# AI OCT Disease Detection and Classification

This repository provides a comprehensive solution for **Optical Coherence Tomography (OCT) Disease Detection and Classification**. The project features a GUI application for detecting retinal diseases using OCT images and a training pipeline for customising and improving the classification models.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
   - [Graphical User Interface (GUI)](#graphical-user-interface-gui)
   - [Model Training](#model-training)
   - [Evaluation](#evaluation)
3. [Architecture and Approach](#architecture-and-approach)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Architectures](#model-architectures)
   - [Class Imbalance Handling](#class-imbalance-handling)
4. [Dataset](#dataset)
5. [Usage and Setup](#usage-and-setup)
6. [Results](#results)
   - [Disease Detection Evaluation](#disease-detection-evaluation)
   - [Disease Classification Evaluation](#disease-classification-evaluation)
   - [Discussion Of Results](#discussion-of-results)
7. [Directory Structure](#directory-structure)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

---

## Introduction

OCT AI is an advanced AI-based framework designed to assist in analyzing Optical Coherence Tomography (OCT) images for detecting and classifying retinal diseases. By leveraging modern deep learning techniques, OCT AI delivers intuitive disease risk assessments and probability predictions for various retinal conditions. This tool aims to empower clinicians and researchers by providing insights from large-scale OCT datasets like the **RFMiD dataset**, while emphasizing its non-diagnostic and research-only nature.

---

## Features

### Graphical User Interface (GUI)

The user-friendly GUI allows for seamless interaction with the OCT AI system. Key features include:

- **Image Upload**: Users can upload an OCT image to analyze.
- **Disease Risk Assessment**: Displays overall risk classification (High Risk or Low Risk).
- **Probability Graph**: A detailed graphical representation of disease probabilities for detected conditions.
- **Summary Insights**: Highlights diseases with significant probabilities alongside their full names and abbreviations.
- **Save Graphs**: Allows saving the generated probability graphs.
- **Help Section**: Offers step-by-step guidance for using the application and explains its functionalities.

---

### Model Training

OCT AI supports training custom models for disease detection and classification. The framework includes:

- Scripts for training both **classification** and **detection** models.
- Class imbalance handling via weighted loss functions and dataset balancing techniques.
- Configurable hyperparameters for batch size, learning rate, epochs, and augmentation techniques.
- Outputs best-performing models based on validation metrics.

---

### Evaluation

Evaluation scripts are provided to measure the performance of trained models. Key evaluation features include:

- **Metrics**: Computes weighted F1 Score, confusion matrices, and loss values.
- **Visualization**: Generates confusion matrices and other visual insights during evaluation.
- Supports both classification and detection evaluations to compare performance across models.

---

## Architecture and Approach

### Data Preprocessing

The preprocessing pipeline is crucial for achieving high-performance model training and includes:

1. **Image Resizing**: All images are resized to `224x224` pixels for model compatibility.
2. **Normalization**: Normalizes pixel values using mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`, compatible with pretrained models.
3. **Augmentation**: Data augmentation techniques applied include:
   - Random rotations and flips.
   - Brightness and contrast adjustments.
   - Elastic transformations and coarse dropout to reduce overfitting.
4. **Class Weights**: Balanced using a weighted loss function to address the class imbalance in diseases with fewer images.

For more details, refer to the dataset exploration notebook in `notebooks/data_exploration.ipynb`.

---

### Model Architectures

#### Classification Model

- **Base Architecture**: A custom convolutional neural network with modifications for multi-label disease classification.
- **Activation Function**: Sigmoid activation for multi-label classification tasks.
- **Loss Function**: Binary Cross-Entropy with logits, weighted by class frequencies.
- **Pretraining**: Supports transfer learning using pretrained backbones like ResNet.

#### Detection Model

- **Objective**: Binary detection model predicting the overall disease risk (High Risk or Low Risk).
- **Layers**: Fully connected layers for binary classification with optimized architectures for OCT images.

---

### Class Imbalance Handling

Class imbalance was mitigated through:
- **Weighted Loss**: Weights inversely proportional to class frequencies.
- **Dataset Reduction**: Diseases with fewer than 100 images were excluded to improve model generalizability and training stability.

---

## Dataset
The dataset used in this project is the publicly available **RFMiD Dataset**, which contains labeled OCT images for multiple retinal diseases.

### Dataset Overview
- **Images**: Captured with specialized fundus cameras with varying resolutions.
- **Labels**: Multi-label classifications for diseases, including common retinal conditions such as diabetic retinopathy (DR) and age-related macular degeneration (ARMD).
- **Structure**:
  - `train`: Contains labeled images for training the models.
  - `val`: Contains labeled images for validation.
  - `test`: Contains labeled images for testing the models.

### Dataset Preprocessing
1. Images are resized to 224x224 pixels for uniformity.
2. Augmentations, including random rotation, flipping, and elastic transformations, are applied to enhance the diversity of training data.
3. Dataset trimming ensures that only diseases with adequate samples are used for training.

### Challenges and Solutions
- **Class Imbalance**: Addressed through weighted loss functions and selective trimming.
- **High Variance in Labels**: Augmentation helps mitigate overfitting and enhances generalizability.

---
## Usage and Setup

The OCT AI system offers multiple functionalities, including a user-friendly GUI for disease detection and a robust training pipeline for customizing classification and detection models.

### Setting Up the Environment

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/OCT-AI.git
   cd OCT-AI
   ```
2. Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # For Linux/MacOS
    .venv\\Scripts\\activate     # For Windows
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
   ```

### Training Models

1. Update dataset paths in the `train_classification_model` and `train_detection_model` functions in the train module.

2. Train the classification model:

    ```bash
    python src/train/train_classification.py
    ````

3. Train the detection model:

    ```bash
    python src/train/train_detection.py
    ```

4. Monitor training outputs:
- Logs with training loss and validation F1 Score.
- Best-performing models saved in the `model` directory.

### Evaluation Scripts

1. Classification evaluation:

    ```bash
    python src/train/evaluate_classification.py
    ```

2. Detection evaluation:

    ```bash
    python src/train/evaluate_detection.py
    ```

3. Outputs:
- Confusion matrices saved as `.png` images.
- Metrics such as weighted F1 Score displayed in the logs.

For a detailed walkthrough, refer to the training and evaluation scripts in the src/train directory.

### Running the GUI

1. Navigate to the GUI directory:

    ```bash
    cd gui
   ```

2. Run the main GUI script:

    ```bash
    python main_gui.py
   ```
   
3. Interact with the application:
- Upload an OCT image. 
- View disease risk and probabilities. 
- Save the probability graph. 
- Access the Help section for guidance.

---

# Results

## Disease Detection Evaluation

The disease detection preformed best with model version 23 achieving the highest overall performance metrics:

- **Accuracy**: 92.66%
- **AUC (Area Under Curve)**: 0.9593
- **Precision**: 0.93 (weighted average)
- **Recall**: 0.93 (weighted average)
- **F1 Score**: 0.93 (weighted average)

### Detailed Classification Report

| Metric      | No Disease | Disease | Macro Avg | Weighted Avg |
|-------------|------------|---------|-----------|--------------|
| Precision   | 0.83       | 0.95    | 0.89      | 0.93         |
| Recall      | 0.82       | 0.95    | 0.89      | 0.93         |
| F1 Score    | 0.82       | 0.95    | 0.89      | 0.93         |
| Support     | 134        | 506     | 648       | 640          |

### Visualizations

#### **Confusion Matrix Heatmap for Disease Detection**:

![confusion_matrix](https://github.com/user-attachments/assets/32cf01c8-cb91-46ff-a871-fe35b82b3a20)

#### **ROC Curve for Disease Detection**:


![roc_curve](https://github.com/user-attachments/assets/c34a4f37-da11-4c61-abc1-aba38719fef0)


---

## Disease Classification Evaluation

The disease classification model performed well across multiple diseases. Below are the overall metrics for model version 15:

- **Accuracy**: 93.46%
- **Precision**: 82.79%
- **Recall**: 83.44%
- **F1 Score**: 83.11%

### Per-Disease Metrics

The classification model achieved the following results for individual diseases:

| Disease                     | F1 Score | Recall | Precision |
|-----------------------------|----------|--------|-----------|
| Diabetic Retinopathy (DR)   | 0.83     | 0.81   | 0.86      |
| Age-Related Macular Degeneration (ARMD) | 0.67 | 0.71 | 0.63      |
| Macular Hole (MH)           | 0.81     | 0.85   | 0.77      |
| Drusen (DN)                 | 0.49     | 0.57   | 0.43      |
| Myopia (MYA)                | 0.78     | 0.78   | 0.78      |
| Tessellation (TSLN)         | 0.60     | 0.58   | 0.62      |
| Optic Disc Cupping (ODC)    | 0.55     | 0.53   | 0.56      |

### Visualizations

#### **Confusion Matrix Heatmaps for Each Disease**:
![TSLN_confusion_matrix](https://github.com/user-attachments/assets/5b01334b-da66-488e-9874-0d2fadab91d7)
![ODC_confusion_matrix](https://github.com/user-attachments/assets/9194adcc-8155-4c4b-aa99-6ff0befbcafe)
![MH_confusion_matrix](https://github.com/user-attachments/assets/dfd371e5-c133-4e4e-a23e-23f95092337f)
![MYA_confusion_matrix](https://github.com/user-attachments/assets/db324551-0fc8-46bd-9a31-16bd5fe1178a)
![DR_confusion_matrix](https://github.com/user-attachments/assets/c64fd41e-64e3-401d-8b61-19dc5f2e7e84)
![DN_confusion_matrix](https://github.com/user-attachments/assets/5f4ced10-c966-4f78-a567-cc90b663d786)
![ARMD_confusion_matrix](https://github.com/user-attachments/assets/0184d4ba-5607-43ca-b23c-bc56fbd91a21)

#### **Overall ROC Curves Including Each Disease**:

![roc_curves_all_diseases](https://github.com/user-attachments/assets/1c0f7c62-8cbe-4549-9d37-e59f7bbb5217)

---

## Discussion of Results

### Strengths:

- The models demonstrated high accuracy and F1 scores across multiple diseases.
- Diseases with sufficient data (e.g., DR and ARMD) achieved high recall and precision values.

### Limitations:

- Diseases with fewer training samples (e.g., DN and ODC) showed lower F1 scores and recall.
- The model's performance was slightly biased toward diseases with larger datasets due to class imbalance.

### Recommendations for Improvement:

- Collect more labeled OCT images for underrepresented diseases to enhance generalization.
- Experiment with advanced data augmentation techniques to mitigate the impact of limited samples.
- Explore transfer learning using larger, pretrained medical imaging models (e.g., Vision Transformers or EfficientNet).

---

## Directory Structure
Below is the current directory structure for this project, please feel free to change this structure in
a way that suits you. 

```
OCT-AI-GUI/
├── README.md               # Project documentation
├── data/                   # Dataset directories
│   ├── classification/     # Classification datasets
│   ├── detection/          # Detection datasets
│   ├── train/              # Training images and labels
│   ├── val/                # Validation images and labels
│   ├── test/               # Testing images and labels
├── gui/                    # GUI scripts and outputs
│   ├── main_gui.py         # Main GUI script
│   └── <image_graphs>.png  # Graphs generated by the GUI
├── model/                  # Trained models
│   ├── <model_name>.pth    # Pretrained model weights
├── notebooks/              # Jupyter notebooks
│   ├── data_exploration.ipynb # Exploratory data analysis
├── requirements.txt        # Required Python dependencies
├── src/                    # Source scripts for training and evaluation
│   ├── data/               # Dataset utilities
│   ├── model/              # Model architectures
│   ├── train/              # Training and evaluation scripts along with evaluation graphs and logs
│   ├── utils/              # Utility scripts
```
---

## Contributing
Contributions are welcome! Please submit a pull request or create an issue. Also feel free to contact any of the 
contributors listed in the [Acknowledgments](#acknowledgments).

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Contributors: **Bilal Fawaz** (*Project Leader, Software Development*), **Ahmed El Ashry** (*Researcher, Consultant*) and **Mahmoud El Ashry** (*Researcher*)
- Dataset: [RFMiD](https://riadd.grand-challenge.org/Download/)
- Libraries: PyTorch, Albumentations, OpenCV, Tkinter, Jupyter, and others.
