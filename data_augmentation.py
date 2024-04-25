import os
import cv2
import shutil
import argparse
from imutils import paths

# Function to apply augmentation and save images
def augment_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each image in the input folder
    for image_path in paths.list_images(input_folder):
        # Read the image
        file_name = os.path.basename(image_path)
        file = os.path.splitext(file_name)
        image = cv2.imread(image_path)
        ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3 = cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
        ret,thresh4 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO)
        ret,thresh5 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO_INV)
        cv2.imwrite(os.path.join(output_folder, file[0] + "_bin" + file[1]), thresh1)
        cv2.imwrite(os.path.join(output_folder, file[0] + "_bininv" + file[1]), thresh2)
        cv2.imwrite(os.path.join(output_folder, file[0] + "_trunc" + file[1]), thresh3)
        cv2.imwrite(os.path.join(output_folder, file[0] + "_tozero" + file[1]), thresh4)
        cv2.imwrite(os.path.join(output_folder, file[0] + "_tozeroinv" + file[1]), thresh5)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Augment images and labels.')
parser.add_argument('input_folder', type=str, help='Path of the folder containing subfolders "images" and "labels".')
parser.add_argument('output_folder', type=str, help='Location where augmented images and labels will be stored.')
args = parser.parse_args()

# Paths for images and labels
images_folder = os.path.join(args.input_folder, 'images')
labels_folder = os.path.join(args.input_folder, 'labels')
output_images_folder = os.path.join(args.output_folder, 'images')
output_labels_folder = os.path.join(args.output_folder, 'labels')

# Augment images
augment_images(images_folder, output_images_folder)

# Copy label files
if not os.path.exists(output_labels_folder):
    os.makedirs(output_labels_folder)

# List all files in the labels folder
label_files = os.listdir(labels_folder)

# Iterate over each label file
for label_file in label_files:
    # Construct source and destination paths
    source_path = os.path.join(labels_folder, label_file)
    for suffix in ["_bin", "_bininv", "_trunc", "_tozero", "_tozeroinv"]:
        # Construct the destination file name with suffix
        dest_filename = label_file.replace('.txt', f'{suffix}.txt')
        dest_path = os.path.join(output_labels_folder, dest_filename)
        # Copy the label file
        shutil.copy(source_path, dest_path)

print("Augmentation and copying completed.")
