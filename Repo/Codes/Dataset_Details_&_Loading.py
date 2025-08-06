import glob
import os

import pandas as pd


def combine_csv_files(input_path, output_path, folder_name):
    """
    Combine all CSV files in a folder into a single CSV file

    Args:
        input_path (str): Path to the folder containing CSV files
        output_path (str): Path where the combined CSV will be saved
        folder_name (str): Name for the output file
    """
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(input_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_path}")
        return

    print(f"Found {len(csv_files)} CSV files in {folder_name}")

    # List to store all dataframes
    dataframes = []

    # Read each CSV file and append to list
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add a column to identify the source file (optional)
            df["source_file"] = os.path.basename(file)
            dataframes.append(df)
            print(f"  - Loaded {file}: {len(df)} rows")
        except Exception as e:
            print(f"  - Error loading {file}: {str(e)}")

    if dataframes:
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Create output filename
        output_file = os.path.join(
            output_path,
            f"combined_{folder_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv",
        )

        # Save combined dataframe
        combined_df.to_csv(output_file, index=False)
        print(f"Combined dataset saved: {output_file}")
        print(
            f"Total rows: {len(combined_df)}, Total columns: {len(combined_df.columns)}"
        )
        print(f"Columns: {list(combined_df.columns)}")
        print("-" * 50)

        return combined_df
    else:
        print(f"No valid dataframes to combine for {folder_name}")


# Main execution
def main():
    # Define paths
    base_input_path = "/kaggle/input/sensordataset-vi-kaj-kor-na-please"
    output_path = "/kaggle/working"  # Kaggle's output directory

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Define subfolders to process
    subfolders = [
        ("Bump (User 17-30)", "Bump_User_17_30"),
        ("Pothole (User 1-16)", "Pothole_User_1_16"),
    ]

    print("Starting CSV combination process...")
    print("=" * 60)

    combined_datasets = {}

    # Process each subfolder
    for folder_display_name, folder_safe_name in subfolders:
        folder_path = os.path.join(base_input_path, folder_display_name)

        if os.path.exists(folder_path):
            print(f"Processing folder: {folder_display_name}")
            combined_df = combine_csv_files(folder_path, output_path, folder_safe_name)
            if combined_df is not None:
                combined_datasets[folder_safe_name] = combined_df
        else:
            print(f"Folder not found: {folder_path}")

    print("=" * 60)
    print("Summary:")
    for name, df in combined_datasets.items():
        print(f"  - {name}: {len(df)} total rows")

    print(f"\nCombined CSV files saved in: {output_path}")


# Run the combination process
if __name__ == "__main__":
    main()
