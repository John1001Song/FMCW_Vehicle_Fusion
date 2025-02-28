import os
import numpy as np
import argparse

def print_npy_info(folder):
    """
    Load each .npy file in the given folder and print its shape.
    """
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    if not files:
        print("No .npy files found in", folder)
    for file in files:
        file_path = os.path.join(folder, file)
        try:
            arr = np.load(file_path, allow_pickle=True)
            print(f"{file}: shape {arr.shape}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Display information for train/val/test npy files in output subfolders."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory containing npy subfolders (e.g., output_npy)")
    args = parser.parse_args()
    
    groups = ["rear_npy", "side_npy", "combined_ego_rear", "combined_ego_side"]
    for group in groups:
        group_folder = os.path.join(args.output_dir, group)
        print(f"--- {group} ---")
        if os.path.isdir(group_folder):
            print_npy_info(group_folder)
        else:
            print(f"Folder {group_folder} does not exist.")
        print()  # Blank line for clarity

if __name__ == "__main__":
    main()
