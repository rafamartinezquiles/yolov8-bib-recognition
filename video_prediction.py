from ultralytics import YOLO
import cv2
import os
import csv
import argparse
import torch

def load_yolo_model(model_path):
    return YOLO(model_path)

def extract_frames(video_path, output_folder, frame_interval):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Variables for frame extraction
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.png")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()

def predict_people(model_people, image_path):
    return model_people.predict(image_path, classes=0)

def process_results_people(results_people, image):
    cropped_images = []

    for result in results_people:
        boxes_people = result.boxes
        for coordenadas in boxes_people.xyxy:
            x_min, y_min, x_max, y_max = map(int, coordenadas)
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_images.append(cropped_image)

    return cropped_images

def predict_rbnr(model_rbnr, cropped_image):
    return model_rbnr.predict(cropped_image)[0]

def get_svhn_number(image, model_svhn):
    results_svhn = model_svhn.predict(image)[0]
    bib_list = []
    if results_svhn.boxes.xyxy.size(0) != 0:
        primeros_elementos = results_svhn.boxes.xyxy[:, 0]
        indices_ordenados = torch.argsort(primeros_elementos)
        class_ordered = results_svhn.boxes.cls[indices_ordenados]
        for number in class_ordered:
            bib_list.append(str(int(number.item())))
        bib = "".join(bib_list)
        return bib
    else:
        return None
    
def write_to_csv(csv_filename, fieldnames, bib_data):
    csv_exists = os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not csv_exists:
            writer.writeheader()

        for bib_entry in bib_data:
            writer.writerow(bib_entry)

def main(args):
    # Load YOLO models
    model_people = load_yolo_model(args.people_model)
    model_rbnr = load_yolo_model(args.bib_model)
    model_svhn = load_yolo_model(args.number_model)

    # Create folder for video frames
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    frames_folder = f"{video_name}_frames"
    extract_frames(args.video_path, frames_folder, args.frame_interval)

    # Process frames for predictions
    bib_data = []
    for frame_file in os.listdir(frames_folder):
        image_path = os.path.join(frames_folder, frame_file)
        results_people = predict_people(model_people, image_path)
        image = cv2.imread(image_path)
        cropped_images = process_results_people(results_people, image)
        for cropped_image in cropped_images:
            results_rbnr = predict_rbnr(model_rbnr, cropped_image)
            if results_rbnr.boxes.xyxy.size(0) != 0:
                x_min, y_min, x_max, y_max = map(int, results_rbnr.boxes.xyxy[0])
                new_image = cropped_image[y_min:y_max, x_min:x_max]
                bib_number = get_svhn_number(new_image, model_svhn)
                bib_data.append({
                    "frame_name": frame_file,
                    "xmin": x_min,
                    "ymin": y_min,
                    "xmax": x_max,
                    "ymax": y_max,
                    "missing_label": bib_number
                })

    # Write predictions to CSV
    pred_csv_file = args.output_csv
    fieldnames = ['frame_name', 'xmin', 'ymin', 'xmax', 'ymax', 'missing_label']
    write_to_csv(pred_csv_file, fieldnames, bib_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict bib numbers from video frames')
    parser.add_argument('people_model', help='Path to the YOLO model to detect people (.pt file)')
    parser.add_argument('bib_model', help='Path to the YOLO model to detect bibs (.pt file)')
    parser.add_argument('number_model', help='Path to the YOLO model to detect numbers (.pt file)')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('frame_interval', type=int, help='Interval (in seconds) between frames to be extracted')
    parser.add_argument('output_csv', help='Output CSV file path')
    args = parser.parse_args()
    main(args)

