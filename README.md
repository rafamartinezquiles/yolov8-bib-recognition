# Assessment and implementation of an object detection framework for bib number recognition during sports events
Using machine learning algorithms, including Ultralytics, Torch, Tensorflow, OpenCV and NIVIDIA's CUDA to detect numbers on racing bibs found in natural image scene. 
![](images/yolo_application.png)

## Overview and Background
Detecting and recognizing Racing Bib Numbers (RBN) entails the challenging tasks of locating the bib attached to a person within a natural scene and deciphering the text inscribed on the bib itself in order to identify the runner. This involves intricate steps such as identifying the bib's area on a person and then interpreting the numerical content. Drawing from prior research and practical experience, this project employs a functional Convolutional Neural Network (CNN) to effectively identify race bib numbers in static images.

This repository delves into the exploration of Convolutional Neural Networks (CNN), particularly focusing on the utilization of Ultralytics (in particular YOLO module) and OpenCV, to discern Racing Bib Numbers (RBNR) within natural image settings. By capitalizing on publicly available and labeled datasets sourced from earlier investigations (refer to the reference section for additional details), results have been achieved in terms of accuracy and prediction time.
![](images/results_accuracy.png)

## Getting started

### Installing
The project is deployed in a local machine, so you need to install the next software and dependencies to start working:

1. Create and activate the new virtual environment for the project

```bash
conda create --name bib_detection python=3.8
conda activate bib_detection
```

2. Clone repository

```bash
git clone https://github.com/rafamartinezquiles/bachelor-s-thesis.git
```

3. In the same folder that the requirements are, install the necessary requirements

```bash
pip install -r requirements.txt
```

### Setup
Once the training data has been downloaded, training can proceed. It is worth noting that the "bib detection big data" provides functionality to download it in YOLOv8 format, which is recommended. Meanwhile, for SVHN data, we would need to download it in format 1 for both the training set and the test set. However, the latter do not come in the desired YOLOv8 format, so preprocessing would be required.

1. Access the directory where the compressed folders associated with the SVHN dataset have been downloaded and execute the following commands to decompress these folders

```bash
tar -xzvf train.tar.gz
tar -xzvf test.tar.gz
```

2. Once the previous folders have been downloaded and unzipped, the next step is to transfer all images to a folder named "images." This is achieved by executing the following commands, where the source path is designated as the first argument and the destination path is the same as the source path, with "/images" appended to ensure the images are saved there. This procedure is applied to both the train and test folders.

```bash
python move_png_files.py /path/to/train_folder/ /path/to/train_folder/images
python move_png_files.py /path/to/test_folder/ /path/to/test_folder/images
```

3. Within the "labels" directory, you'll encounter two subdirectories: "labels_train" and "labels_test." Place these folders inside the "train" and "test" directories, respectively, so that each contains both "images" and "labels" directories. To accomplish this, rename "labels_train" and "labels_test" to simply "labels" within their respective folders.

## Data Details

### Training
- [Bib Detection Big Data](https://universe.roboflow.com/hcmus-3p8wh/bib-detection-big-data).
- [Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers)

### Testing
- [Trans Gran Canaria Race Bib Number in the Wild (TGCRBNW)](http://hdl.handle.net/10553/112156).
- [Racing Bib Number Recognition (RBNR)](https://people.csail.mit.edu/talidekel/RBNR.html).

## References
- OpenCV: https://opencv.org/
- Ultralytics: https://github.com/ultralytics/ultralytics
- HCMUS. bib detection big data dataset. https://universe.roboflow.com/hcmus-3p8wh/bib-detection-big-data, jun 2023. visited on 2024-02-12
- Li Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6):141–142, 2012.
- Pablo Hernández-Carrascosa, Adrian Penate-Sanchez, Javier Lorenzo-Navarro, David Freire-Obregón, and Modesto Castrill ́on-Santana. Tgcrbnw: A dataset for runner bib number detection (and recognition) in the wild. In 2020 25th International Conference on Pattern Recognition (ICPR), pages 9445–9451, 2021.
- I.B. Ami, T. Basha, and S. Avidan. Racing bib number recognition. In Proc. BMCV,pages 1–10, 2012



