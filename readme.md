# MNIST Digit Classification Project

## Overview
This project involves building, training, and evaluating a machine learning model to recognize handwritten digits from the popular MNIST dataset. The MNIST dataset is widely used as a benchmark for image processing algorithms and machine learning models, making it an ideal project for practicing and understanding the fundamentals of deep learning and computer vision.

## Project Structure
- `data/`: Contains the training and testing datasets. link: https://www.kaggle.com/competitions/digit-recognizer/data
- `notebooks/`: Jupyter notebooks that detail the data exploration, preprocessing, model building, and evaluation processes.
- `README.md`: This file, providing an overview of the project.

## Key Features
-**Importing Packages** : All the necessary packages for this project.
- **Data Preprocessing**: Loading, Normalization, splitting to train and Valadition datasets and reshaping of image data for model input.
- **Model Architecture**: keras sequential model built using frameworks  TensorFlow.
- **Training and Evaluation**: Detailed training process with metrics like accuracy and loss tracked over epochs.
- **Visualization**: Plots of training history and model predictions to aid in performance evaluation.

## Installation and Setup
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/mnist-digit-classification.git
    cd mnist-digit-classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook or script:
    ```bash
    jupyter notebook notebooks/mnist_classification.ipynb
    ```

## Dataset
The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits, each image being a 28x28 grayscale image labeled from 0 to 9.

## Model Details
- **Architecture**: A simple Convolutional Neural Network (CNN) or a fully connected neural network.
- **Layers**: Includes convolutional, activation (ReLU), and dense layers.
- **Optimizer**:  Adam optimizer for efficient training.
- **Loss Function**: SparseCategoricalCrossentropy for multi-class classification.

## Results
- **Accuracy**: Achieved an accuracy of 97% on the test set.
- **Loss**: Final loss on the validation set was 0.0928

## Visualization
- Training and validation accuracy/loss plots.
- Sample predictions with actual vs. predicted labels.


