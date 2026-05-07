import numpy as np
from PIL import Image
import streamlit as st
 
# These imports assume the CNN source files are in the same directory
from denselayer import Layer_Dense
from ConvolutionLayer import Convolution
from Relu import ReLu_Activation
from maxpool import MaxPool
from flatten_layer import Flatten
 
 
@st.cache_resource
def load_model():
    """Initialize architecture and load trained weights. Cached so it runs once."""
    convolution1 = Convolution(num_filters=8,  filter_size=3, input_depth=1,  stride=1, padding=1)
    convolution2 = Convolution(num_filters=16, filter_size=3, input_depth=8,  stride=1, padding=1)
    relu1    = ReLu_Activation()
    relu2    = ReLu_Activation()
    maxpool1 = MaxPool(pool_size=2, stride=2)
    maxpool2 = MaxPool(pool_size=2, stride=2)
    flatten1 = Flatten()
 
    # Dummy forward pass to infer dense layer input size
    dummy = np.random.randn(1, 1, 28, 28)
    flat_out = flatten1.forward(
        maxpool2.forward(
            relu2.forward(
                convolution2.forward(
                    maxpool1.forward(
                        relu1.forward(
                            convolution1.forward(dummy)
                        )
                    )
                )
            )
        )
    )
    n_inputs = flat_out.shape[1]
    hidden_dense = Layer_Dense(n_inputs, n_neurons=64)
    dense1       = Layer_Dense(64, n_neurons=10)
 
    # Load weights
    data = np.load('mnist_model.npz')
    convolution1.filter  = data['conv1_filters']
    convolution1.biases  = data['conv1_biases']
    convolution2.filter  = data['conv2_filters']
    convolution2.biases  = data['conv2_biases']
    hidden_dense.weights = data['hidden_dense_weights']
    hidden_dense.biases  = data['hidden_dense_biases']
    dense1.weights       = data['dense1_weights']
    dense1.biases        = data['dense1_biases']
 
    return convolution1, relu1, maxpool1, convolution2, relu2, maxpool2, flatten1, hidden_dense, dense1
 
 
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
 
 
def preprocess(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        processed  : (1, 1, 28, 28) float32 array ready for inference
        thumb      : (28, 28) uint8 array for display
    """
    img = img.convert('L').resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = 1.0 - arr                             # invert: model trained on white-on-black
    thumb = (arr * 255).astype(np.uint8)
    return arr.reshape(1, 1, 28, 28), thumb
 
 
def predict(img_array: np.ndarray) -> np.ndarray:
    """Run forward pass, return (10,) probability array."""
    conv1, relu1, pool1, conv2, relu2, pool2, flat, hidden, out = load_model()
    x = conv1.forward(img_array)
    x = relu1.forward(x)
    x = pool1.forward(x)
    x = conv2.forward(x)
    x = relu2.forward(x)
    x = pool2.forward(x)
    x = flat.forward(x)
    x = hidden.forward(x)
    x = out.forward(x)
    return softmax(x)[0]          # shape (10,)



