# COCO Detection with TensorFlow: SSD MobileNet v2

This project demonstrates object detection using a pre-trained SSD MobileNet v2 model on the COCO dataset with TensorFlow. The model is used for inference on unseen images, showcasing the capabilities of Single Shot MultiBox Detector (SSD) and MobileNet v2 to perform fast and accurate object detection.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Machine Learning Techniques](#machine-learning-techniques)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Overview

The goal of this project is to detect objects from the COCO dataset using a pre-trained model. The project utilizes a TensorFlow implementation of SSD (Single Shot MultiBox Detector) with MobileNet v2 as the backbone for feature extraction. This setup is ideal for real-time applications due to its balance between speed and accuracy.

The notebook provides a step-by-step guide to:

1. Load a pre-trained SSD MobileNet v2 model.
2. Create a label map for COCO dataset classes.
3. Perform inference on unseen images.
4. Visualize the results.

## Dataset

The COCO dataset is a large-scale object detection, segmentation, and captioning dataset. In this project, we utilize the label map corresponding to the COCO dataset to identify the detected objects. The label map is used to convert class indices into human-readable class names during inference.

## Model Architecture

### SSD MobileNet v2

- **Single Shot MultiBox Detector (SSD)**: SSD is a popular object detection algorithm that discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. During prediction, SSD generates category scores for the presence of objects in each default box and makes adjustments to the box to better match the object shape.
  
- **MobileNet v2**: MobileNet v2 is used as the backbone network for feature extraction in this project. MobileNet v2 is a lightweight, efficient deep neural network designed for mobile and embedded vision applications. It uses depthwise separable convolutions to reduce the number of parameters and computations while maintaining accuracy.

### Key Components

1. **Base Network**: MobileNet v2 serves as the base network to extract feature maps from the input image. These feature maps are used by SSD to make predictions at multiple scales.
  
2. **Detection Heads**: The SSD model has detection heads attached to multiple layers of the base network to predict object classes and bounding boxes at different scales, allowing for effective detection of small and large objects.

3. **Non-Maximum Suppression (NMS)**: NMS is applied to remove redundant bounding boxes with a high overlap, ensuring that each object is detected only once.

## Machine Learning Techniques

1. **Transfer Learning**: The model used in this project is pre-trained on the COCO dataset. Transfer learning allows us to leverage the knowledge gained by the model during pre-training, significantly reducing the training time and computational requirements needed to perform accurate detections.

2. **Convolutional Neural Networks (CNNs)**: The backbone of the SSD model, MobileNet v2, is a type of Convolutional Neural Network. CNNs are ideal for processing image data due to their ability to learn spatial hierarchies of features.

3. **Data Preprocessing**: Images are preprocessed to match the input requirements of the SSD MobileNet v2 model. This includes resizing, normalization, and conversion to tensors.

4. **Bounding Box Regression**: The model predicts bounding box coordinates in addition to class labels. Bounding box regression is used to refine these coordinates, ensuring the predicted boxes closely match the actual object boundaries.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV

### Installation

Clone the repository:

```sh
git clone https://github.com/your-username/coco-detection-tensorflow.git
cd coco-detection-tensorflow
```

Install the required packages:

```sh
pip install -r requirements.txt
```

## Usage

1. **Create Label Map**: The label map for the COCO dataset is used to map class indices to class names.

2. **Load the Pre-trained Model**: Load the SSD MobileNet v2 model from the TensorFlow Model Zoo or the provided link to pre-trained TensorRT engines.

3. **Perform Inference**: Use the model to perform inference on unseen images. The notebook provides a script to load an image, run inference, and visualize the detected objects.

4. **Visualize Results**: Bounding boxes, class labels, and confidence scores are plotted on the image using Matplotlib or OpenCV for visualization.

## Results

The SSD MobileNet v2 model performs well on a variety of objects from the COCO dataset. It provides real-time object detection capabilities, making it suitable for applications such as video surveillance, autonomous driving, and robotics.

### Example Output

The following image demonstrates the output of the model, showing detected objects with bounding boxes and class labels:

<img width="1038" alt="image" src="https://github.com/user-attachments/assets/8371761d-9dab-43ac-87c8-41ead3783d5d">


## References

- [TensorFlow Model Zoo](https://www.tensorflow.org/lite/guide/model_zoo)
- [COCO Dataset](https://cocodataset.org/)
- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
- [MobileNet v2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
