# TFG-Real_Time_Facial_Recognition

## Introduction

This repository is the practical part of my Final Degree Project in Computer Engineering. It hosts an exploration of the field of facial recognition using deep learning. Inspired by a crossover interest in machine learning and the visual world, this project investigates the complexities of facial recognition, focusing on model performance, speed, and efficiency in real-time implementation.

## Models and Techniques

We have compared between two types of loss functions and different models of the Convolutional Neural Network.

```python
models = [
   "ResNet50",
  "MobileNetV3",
  "InceptionV3",
  "EfficientNet", 
  "VGG16"
]

loss_functions = [
   "Triplet Loss",
  "Categorical Cross Entropy",
]

face_detector_alignment_crop = ["MTCNN"]
face_detector_real_time = ["Media-pipe"]

```
## Evaluation

| Base Model | Accuracy | Precision |Recall |F1-Score |
| ---   | --- | --- |--- |--- |
| ResNet50 - Triplet Loss | **97,78%** | **97,62%** |**97,78%** |**97,62%** |
| ResNet50 - Categorical Cross Entropy | 80% | 82,86% |80,88% |80,02% |
| MobileNetV3 - Triplet Loss | 91,11% | 92,98% |92,16% |91,63% |
| MobileNetV3 - Categorical Cross Entropy | 75,55% | 77,73% | 74,45%| 74,74%|
| InceptionV3 - Triplet Loss | 64,44% |64,01%|66,19%|64,41%|
| InceptionV3 - Categorical Cross Entropy | 66,66%|70,36%|66,00%|66,97%|
| EfficientNet - Triplet Loss | 95,55%|95,56%|95,82%|95,46%
| VGG16 - Categorical Cross Entropy | 84,44%|84,44%|84,81%|84,53%|

## Data Visualization

In the files triplet_loss_val.ipynb and cross_entropy_val.ipynb we can visualize the results obtained with the test dataset

- Prediction on test dataset - ResNet50 - Triplet Loss
  
![alt text](https://github.com/sergisolis/TFG-Real_Time_Facial_Recognition/blob/main/imgs/triplet_loss_test/ResNet50.png?raw=true)


- Confusion Matrix - ResNet50 - Triplet Loss
  
![alt text](https://github.com/sergisolis/TFG-Real_Time_Facial_Recognition/blob/main/imgs/triplet_loss_test/Resnet50-Confusion-Matrix.png)

## Real-time Deployment

To test models in real time we change the model path based on the missing function that we want to use in the real_time.ipynb file.

Below we have a sample of a recognition frame in real time with the model that gave us the best result (ResNet50 with the use of Triplet Loss):

![alt text](https://github.com/sergisolis/TFG-Real_Time_Facial_Recognition/blob/main/imgs/real-time_frame.png)


## References

- DeepFace: https://github.com/serengil/deepface
- Keras-FaceNet: https://github.com/nyoki-mtl/keras-facenet
- Face Recognition: Siamese w/ Triplet loss: https://www.kaggle.com/code/stoicstatic/face-recognition-siamese-w-triplet-loss
- Facenet: https://arxiv.org/abs/1503.03832
