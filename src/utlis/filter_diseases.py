import pandas as pd

def main():
    # Threshold for minimum number of samples per disease
    MIN_SAMPLES = 100

    # Paths to the CSV files
    train_csv = "../../data/train/RFMiD_Training_Labels.csv"
    val_csv = "../../data/val/RFMiD_Validation_Labels.csv"
    test_csv = "../../data/test/RFMiD_Testing_Labels.csv"

    # Output paths for filtered datasets
    filtered_train_csv = "../../data/train/RFMiD_Training_Labels_Filtered.csv"
    filtered_val_csv = "../../data/val/RFMiD_Validation_Labels_Filtered.csv"
    filtered_test_csv = "../../data/test/RFMiD_Testing_Labels_Filtered.csv"

    # Load the datasets
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Drop diseases with fewer than MIN_SAMPLES
    label_columns = train_df.columns[1:]  # Assuming first column is "ID"
    disease_counts = train_df[label_columns].sum()
    valid_diseases = disease_counts[disease_counts >= MIN_SAMPLES].index

    print(f"Keeping the following diseases with >= {MIN_SAMPLES} samples: {valid_diseases.tolist()}")

    # Filter the datasets to keep only valid diseases
    train_df_filtered = train_df[["ID"] + valid_diseases.tolist()]
    val_df_filtered = val_df[["ID"] + valid_diseases.tolist()]
    test_df_filtered = test_df[["ID"] + valid_diseases.tolist()]

    # Save the filtered datasets
    train_df_filtered.to_csv(filtered_train_csv, index=False)
    val_df_filtered.to_csv(filtered_val_csv, index=False)
    test_df_filtered.to_csv(filtered_test_csv, index=False)

    print("Filtered datasets saved successfully.")

if __name__ == "__main__":
    main()
