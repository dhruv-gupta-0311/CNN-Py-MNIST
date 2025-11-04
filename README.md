CNN-Py-MNIST 
A convolutional neural network built from scratch in Python/NumPy for handwritten digit recognition (MNIST).

Overview:

Implements a CNN architecture using only NumPy (no high-level deep learning libraries) for educational purposes.

Demonstrates key deep learning concepts: convolutional layers, pooling, activation functions, flattening, dense layers, forward/backward propagation, cross-entropy loss, and optimization.

Project originally built for the MNIST dataset and later extended for potential civic-issue image identification.

Features:

-> Custom built layers: Convolution, MaxPooling, Dense (fully connected).

-> Activation functions: ReLU, Softmax.

-> Loss function: Categorical Cross-Entropy.

-> Training loop with manual weight & bias updates.

-> Prediction script to infer digit class from image input.

-> Simple preprocessing pipeline: normalization and reshaping.

Usage:

-> Pre-installed Python, TensorFlow, MNIST dataset, NumPy.

-> For training set Learning rate, Epoches, Batch size.

-> Parameters after training stored in npz file.

-> For prediction, load predict.py.

-> The script loads image, preprocess it, then output the predicted class with accuracy/probability.

Project Architecture:

ConvolutionLayer.py – convolution layer implementation (forward & backward).

maxpool.py – max-pooling layer implementation.(Takes a 2x2 tensor and only maximum value is output)

flatten_layer.py – flattening layer.

denselayer.py – fully-connected (dense) layer.()

Relu.py – ReLU activation implementation.(returns [0 when x -ve, x +ve]; where x is the value inputed)

Softmax_Loss.py – Softmax activation combined with cross-entropy loss.(Calculate probability, loss by taking dense layer output)

train.py – main training loop script.

predict.py – image inference script.

mnist_model.npz – example saved model weights 

images/ – sample digit or additional test images.

Future Work

Extend to civic issue detection (potholes, garbage, broken streetlights) using a real-world dataset.



