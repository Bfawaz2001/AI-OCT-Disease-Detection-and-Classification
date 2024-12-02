import os
import pandas as pd

def main():
    # Define paths
    data_dir = "../../data"
    image_dirs = {
        "train": os.path.join(data_dir, "train", "images"),
        "val": os.path.join(data_dir, "val", "images"),
        "test": os.path.join(data_dir, "test", "images")
    }

    csv_files = {
        "train": os.path.join(data_dir, "train", "RFMiD_Training_Labels.csv"),
        "val": os.path.join(data_dir, "val", "RFMiD_Validation_Labels.csv"),
        "test": os.path.join(data_dir, "test", "RFMiD_Testing_Labels.csv")
    }

    # Update CSV files
    for split, csv_file in csv_files.items():
        df = pd.read_csv(csv_file)
        image_dir = image_dirs[split]
        df["path"] = df["ID"].apply(lambda x: os.path.join(image_dir, f"{x}.png"))  # Adjust file extension if needed
        df.to_csv(csv_file, index=False)
        print(f"Updated {split} CSV with paths.")

if __name__ == "__main__":
    main()
