# Martian Frost Detection Using CNN and Transfer Learning

This project involves building a classifier to distinguish between frosted and non-frosted Martian terrain in HiRISE images using convolutional neural networks (CNN) and transfer learning. The project focuses on data augmentation, regularization techniques, and the application of pre-trained models as feature extractors. This project is part of the final project for the DSCI 552 course.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [CNN Model](#cnn-model)
- [Transfer Learning](#transfer-learning)
- [Evaluation](#evaluation)
- [Requirements](#requirements)

## Project Overview
The main objectives of this project are:
1. To build a CNN model to classify frost in HiRISE images from Mars.
2. To enhance the model's performance using data augmentation and transfer learning with pre-trained networks.
3. To evaluate and compare the CNN model with transfer learning models.

## Dataset
The **Mars Frost HiRISE Dataset** is available from the [NASA Dataverse Repository](https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=doi:10.48577/jpl.QJ9PYA). The dataset consists of 214 subframes with 119,920 labeled 299x299 pixel tiles, categorized as ‘frost’ or ‘background.’

Data is divided into train, test, and validation sets, as specified in:
- `train source images.txt`
- `test source images.txt`
- `val source images.txt`

## CNN Model
1. **Image Augmentation**:
   - Images are augmented with random cropping, zooming, rotation, flipping, contrast adjustment, and translation.
2. **Architecture**:
   - A three-layer CNN is followed by a dense layer. ReLU activations are used in all layers, with a softmax output layer.
   - Techniques such as batch normalization, dropout (30%), and L2 regularization are used.
3. **Training**:
   - The model is trained using cross-entropy loss and the ADAM optimizer for at least 20 epochs, with early stopping based on validation loss.
4. **Evaluation**:
   - Precision, Recall, and F1-score are reported, and training and validation errors are plotted against epochs.

## Transfer Learning
1. **Pre-trained Models**:
   - The models EfficientNetB0, ResNet50, and VGG16 are used for transfer learning.
2. **Implementation**:
   - The last fully connected layer of each model is trained, with all other layers frozen. Features from the penultimate layer serve as input for the new layer.
3. **Training and Evaluation**:
   - Similar to the CNN model, image augmentation and batch normalization are applied. The model is trained for at least 10 epochs, with early stopping based on validation error.
   - Performance metrics and training/validation error plots are generated.

## Evaluation
- Precision, Recall, and F1-score are reported for each model.
- A comparison between the CNN model and transfer learning models is provided to analyze the impact of using pre-trained networks.

## Requirements
The project requires:
- Python
- Libraries: `numpy`, `pandas`, `matplotlib`, `tensorflow` (with Keras), `opencv-python`