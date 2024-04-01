# Detecting Leukemia Using Convolutional Neural Network

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/downloads/)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras Version](https://img.shields.io/badge/Keras-2.x-red)](https://keras.io/)

## Overview

This project implements a Convolutional Neural Network (CNN) to detect leukemia from medical images.
To implement a machine learning model using a CNN for this purpose, you can follow these steps:

**Data Collection and Preprocessing:**
- Gather a dataset of medical images containing both leukemia-positive and leukemia-negative samples.
- Preprocess the images by resizing, normalizing, and augmenting them to ensure uniformity and enhance model performance.

**Model Architecture Design:**
- Define the architecture of the CNN model, including the number of convolutional layers, pooling layers, activation functions, and output layers.
- Consider using pre-trained models like VGG, ResNet, or custom architectures based on the complexity of the task.

**Model Compilation:**
- Compile the CNN model by specifying the optimizer, loss function (e.g., binary cross-entropy for binary classification), and evaluation metrics (e.g., accuracy, precision, recall).

**Model Training:**
- Split the dataset into training and validation sets.
- Train the CNN model on the training data using techniques like mini-batch gradient descent and backpropagation.
- Monitor the training process for metrics like loss and accuracy to ensure the model is learning effectively.

**Model Evaluation:**
- Evaluate the trained model on the test dataset to assess its performance on unseen data.
- Calculate metrics such as accuracy, precision, recall, and F1-score to measure the model's effectiveness in leukemia detection.

**Model Deployment:**
- Once satisfied with the model's performance, deploy it for real-world use, such as integrating it into a web application or healthcare system for automated leukemia detection.

By following these steps and leveraging the capabilities of CNNs, you can build a robust machine learning model for detecting leukemia from medical images, contributing to advancements in medical diagnostics and healthcare technology.


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/Detecting-Leukemia-Using-Convolutinal-Neural-Network.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Detecting-Leukemia-Using-Convolutinal-Neural-Network
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Download the dataset and preprocess it (if needed).
2. Train the model using `train.py`.
3. Evaluate the model using `evaluate.py`.
4. Use the trained model for inference.

## Data

The dataset consists of labeled medical images of leukemia cells. It can be downloaded from [source](https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data).

## Model Architecture

The CNN architecture consists of several convolutional and pooling layers followed by fully connected layers.
Well, a Convolutional Neural Network, or CNN for short, works kind of like that! But instead of a puzzle, we're dealing with images.

First, let's talk about the different parts of a CNN:

- **Convolutional Layer:** This is like a little magnifying glass that scans the image, looking for important features like edges, shapes, or patterns. It does this by applying filters to small sections of the image and moving across it step by step.

- **Pooling Layer:** After each convolutional layer, we use a pooling layer to shrink down the information. It's like zooming out of the puzzle to see the bigger picture. Pooling helps reduce the complexity of the model and makes it easier to process.

- **Activation Function:** This is like a switch that decides whether a neuron should be "on" or "off" based on the information it receives. It adds non-linearity to the model, allowing it to learn more complex patterns.

- **Fully Connected Layer:** Once we've looked at all the pieces of the puzzle and found the important features, we put them together in a fully connected layer. This layer connects every neuron from the previous layer to every neuron in the next layer, helping us make sense of the information.

So, putting it all together, a CNN takes an image as input, passes it through multiple layers of convolutions, pooling, and activations to extract important features, and finally, uses fully connected layers to make predictions.

Think of it like solving a big jigsaw puzzle – one piece at a time – until you reveal the whole picture!


## Training

Train the model by running:

```bash
python train.py
```
**Steps Involved:**

1. **Data Preparation:** The `train.py` script loads the training dataset and preprocesses it as needed. This may involve tasks such as resizing, normalization, and augmentation to ensure the model receives high-quality input data.

2. **Model Definition:** We define the architecture of our CNN model within the `train.py` script. This includes specifying the number of convolutional layers, pooling layers, activation functions, and output layers. We may also use pre-trained models like VGG or ResNet, depending on the complexity of the task.

3. **Model Compilation:** Before training, we compile the model by specifying the optimizer, loss function, and evaluation metrics. For example, we may use the Adam optimizer, binary cross-entropy loss for binary classification, and accuracy as the evaluation metric.

4. **Model Training:** With the dataset prepared and the model compiled, we proceed to train the model on the training data. This involves feeding batches of images into the model, computing the loss, and updating the model's weights using backpropagation. The training process continues for a specified number of epochs.

5. **Monitoring Training Progress:** Throughout the training process, we monitor various metrics such as loss and accuracy to evaluate the model's performance. We may also use callbacks to implement early stopping or save the best model weights.

6. **Saving the Model:** Once training is complete, we save the trained model's weights to disk for future use. This allows us to load the model and make predictions without having to retrain it from scratch.


# Credit

This project is credited to:

- [Vishesh Yadav](https://github.com/vishesh9131) 

## References

Here are some potential references related to "Detecting Leukemia Using Convolutional Neural Network":

- **Paper:**
  - Title: [Leukemia classification using the deep learning method of CNN](https://pubmed.ncbi.nlm.nih.gov/35253723/)
    - This study explores using ResNet-34 and DenseNet-121 architectures for leukemia classification.

- **Paper:**
  - Title: [A Deep Learning Framework for Leukemia Cancer Detection in Microscopic Blood Samples Using Squeeze and Excitation Learning](https://arxiv.org/abs/2004.14866](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.researchgate.net/publication/358253652_A_Deep_Learning_Framework_for_Leukemia_Cancer_Detection_in_Microscopic_Blood_Samples_Using_Squeeze_and_Excitation_Learning&ved=2ahUKEwixnqyRwaGFAxWG1TgGHY1oCz4QFnoECBUQAQ&usg=AOvVaw2gLhjMuwmY_TBMEYGmv2Ea)
    - This paper discusses transfer learning with CNNs for leukemia detection in blood samples.

- **Article:**
  - Leukemia Blood Cell Image Classification Using Convolutional Neural Network
    - (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0213626#:~:text=The%20CNN%20was%20built%20according,and%202%20fully%20connected%20layers).


