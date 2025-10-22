# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np  
import tensorflow as tf
from denselayer import Layer_Dense
from ConvolutionLayer import Convolution
from Relu import ReLu_Activation
from maxpool import MaxPool
from flatten_layer import Flatten
from PIL import Image

convolution1 = Convolution(num_filters=8, filter_size=3, input_depth=1, stride=1, padding=1)
convolution2 = Convolution(num_filters=16, filter_size=3, input_depth=8, stride=1, padding=1)
relu1 = ReLu_Activation()
relu2 = ReLu_Activation()
maxpool1 = MaxPool(pool_size=2, stride=2)
maxpool2 = MaxPool(pool_size=2, stride=2)
flatten1 = Flatten()
dummy_input = np.random.randn(1, 1, 28, 28)
conv_out = convolution1.forward(dummy_input)
relu_out = relu1.forward(conv_out)
pool_out = maxpool1.forward(relu_out)
conv2_out = convolution2.forward(pool_out)
relu2_out = relu2.forward(conv2_out)
pool2_out = maxpool2.forward(relu2_out)
flat_out = flatten1.forward(pool2_out)
n_inputs = flat_out.shape[1]
hidden_dense = Layer_Dense(n_inputs, n_neurons=64)  
dense1 = Layer_Dense(64, n_neurons=10)

print("loading model...")
model_data = np.load('mnist_model.npz')
convolution1.filter = model_data['conv1_filters']
convolution1.biases = model_data['conv1_biases']
convolution2.filter = model_data['conv2_filters']
convolution2.biases = model_data['conv2_biases']
dense1.weights = model_data['dense1_weights']
dense1.biases = model_data['dense1_biases']
hidden_dense.weights = model_data['hidden_dense_weights']
hidden_dense.biases = model_data['hidden_dense_biases']
print("Model loaded successfully.")

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.reshape(-1, 1, 28, 28)
    img_array = 1 - img_array
    return img_array
  

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

image_path = r'images\1002.png'
conv_output = convolution1.forward(preprocess_image(image_path))
relu_output = relu1.forward(conv_output)
pool_output = maxpool1.forward(relu_output)
conv_2 = convolution2.forward(pool_output)
relu_output2 = relu2.forward(conv_2)
pool_output2 = maxpool2.forward(relu_output2)
flat_output = flatten1.forward(pool_output2)
dense_output_hidden = hidden_dense.forward(flat_output)
dense_output = dense1.forward(dense_output_hidden)
probability = softmax(dense_output)
predictions = np.argmax(probability, axis=1)
for i, p in enumerate(probability[0]):
    print(f"Class {i}: {p*100:.2f}%")
print(f"Predicted class for the input image: {predictions[0]}")



