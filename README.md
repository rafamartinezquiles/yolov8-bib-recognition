# Assessment and implementation of an object detection framework for bib number recognition during sports events
Using machine learning algorithms, including Ultralytics, Torch, Tensorflow, OpenCV and NIVIDIA's CUDA to detect numbers on racing bibs found in natural image scene. 

### Overview and Background
Detecting and recognizing Racing Bib Numbers (RBN) entails the challenging tasks of locating the bib attached to a person within a natural scene and deciphering the text inscribed on the bib itself in order to identify the runner. This involves intricate steps such as identifying the bib's area on a person and then interpreting the numerical content. Drawing from prior research and practical experience, this project employs a functional Convolutional Neural Network (CNN) to effectively identify race bib numbers in static images.

This repository delves into the exploration of Convolutional Neural Networks (CNN), particularly focusing on the utilization of Ultralytics (in particular YOLO module) and OpenCV, to discern Racing Bib Numbers (RBNR) within natural image settings. By capitalizing on publicly available and labeled datasets sourced from earlier investigations (refer to the reference section for additional details), results have been achieved in terms of accuracy and prediction time.

### Data Details (Training)
- [Bib Detection Big Data](https://universe.roboflow.com/hcmus-3p8wh/bib-detection-big-data).
- [Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers)

### Data Details (Testing)
- [Trans Gran Canaria Race Bib Number in the Wild (TGCRBNW)](http://hdl.handle.net/10553/112156).
- [Racing Bib Number Recognition (RBNR)](https://people.csail.mit.edu/talidekel/RBNR.html).

### References
- OpenCV: https://opencv.org/
- Ultralytics: https://github.com/ultralytics/ultralytics
- HCMUS. bib detection big data dataset. https://universe.roboflow.com/hcmus-3p8wh/bib-detection-big-data, jun 2023. visited on 2024-02-12
- Li Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6):141–142, 2012.
- Pablo Hernández-Carrascosa, Adrian Penate-Sanchez, Javier Lorenzo-Navarro, David Freire-Obregón, and Modesto Castrill ́on-Santana. Tgcrbnw: A dataset for runner bib number detection (and recognition) in the wild. In 2020 25th International Conference on Pattern Recognition (ICPR), pages 9445–9451, 2021.
- I.B. Ami, T. Basha, and S. Avidan. Racing bib number recognition. In Proc. BMCV,pages 1–10, 2012



