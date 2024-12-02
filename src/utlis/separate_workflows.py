import pandas as pd


def main():
    # Paths to filtered datasets
    train_csv = "../../data/train/RFMiD_Training_Labels_Filtered.csv"
    val_csv = "../../data/val/RFMiD_Validation_Labels_Filtered.csv"
    test_csv = "../../data/test/RFMiD_Testing_Labels_Filtered.csv"

    # Output paths
    detection_train_csv = "../../data/detection/RFMiD_Training_Detection.csv"
    detection_val_csv = "../../data/detection/RFMiD_Validation_Detection.csv"
    detection_test_csv = "../../data/detection/RFMiD_Testing_Detection.csv"

    classification_train_csv = "../../data/classification/RFMiD_Training_Classification.csv"
    classification_val_csv = "../../data/classification/RFMiD_Validation_Classification.csv"
    classification_test_csv = "../../data/classification/RFMiD_Testing_Classification.csv"

    # Columns for each workflow
    detection_columns = ["ID", "Disease_Risk"]
    classification_columns = ["ID", "DR", "ARMD", "MH", "DN", "MYA", "TSLN", "ODC"]

    # Load and split datasets
    for csv_path, detection_path, classification_path in [
        (train_csv, detection_train_csv, classification_train_csv),
        (val_csv, detection_val_csv, classification_val_csv),
        (test_csv, detection_test_csv, classification_test_csv),
    ]:
        df = pd.read_csv(csv_path)
        df[detection_columns].to_csv(detection_path, index=False)
        df[classification_columns].to_csv(classification_path, index=False)

    print("Datasets for Disease Detection and Classification created successfully.")


if __name__ == "__main__":
    main()