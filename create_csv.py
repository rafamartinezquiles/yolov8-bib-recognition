import os
import csv
import argparse

def create_csv(folder_path):
    # Iterate through each subfolder
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        # Check if the item is a directory
        if os.path.isdir(subdir_path):
            # Check if list.txt exists in the directory
            list_file = os.path.join(subdir_path, "list.txt")
            if os.path.exists(list_file):
                csv_filename = os.path.join(subdir_path, f"{subdir}.csv")
                with open(list_file, 'r') as file:
                    lines = file.readlines()
                    # Ignore the first line and remove the last two lines
                    data = [line.strip().split() for line in lines[1:-2]]
                    header = ["Image", "RBN"]
                    with open(csv_filename, 'w', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(header)
                        writer.writerows(data)
                        print(f"CSV file created for {subdir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create CSV files from list.txt in each subfolder.')
    parser.add_argument('folder_path', type=str, help='Path to the uncompressed folder')
    args = parser.parse_args()

    # Check if the specified folder exists
    if not os.path.exists(args.folder_path):
        print("Error: The specified folder does not exist.")
        return

    # Create CSV files
    create_csv(args.folder_path)

if __name__ == "__main__":
    main()

