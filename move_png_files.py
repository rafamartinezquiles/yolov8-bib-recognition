import os
import sys
import shutil

def move_png_files(source_folder, destination_folder):
    # Check if source folder exists
    if not os.path.exists(source_folder):
        print("Source folder does not exist.")
        return

    # Check if destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through files in source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".png"):
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            # Move the file to destination folder
            shutil.move(source_file, destination_file)
            print(f"Moved {filename} to {destination_folder}")

if __name__ == "__main__":
    # Check if correct number of arguments provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <source_folder> <destination_folder>")
        sys.exit(1)

    source_folder = sys.argv[1]
    destination_folder = sys.argv[2]

    move_png_files(source_folder, destination_folder)
