import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from denselayer import Layer_Dense
from ConvolutionLayer import Convolution

from Relu import ReLu_Activation
from Softmax_Loss import SoftMax_Crossentropy
from maxpool import MaxPool
from flatten_layer import Flatten


(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
num_classes = len(np.unique(Y_train))  
# test on sample input to verify components are working
# X_sample = np.random.randn(2, 1, 5, 5)
# Y_sample = np.array([0, 2])
X_sample = X_train[:5]
Y_sample = Y_train[:5]
convolution1 = Convolution(num_filters=2, filter_size=3, input_depth=1, stride=1, padding=1)
relu1 = ReLu_Activation()
maxpool1 = MaxPool(pool_size=2, stride=2)
flatten1 = Flatten()

softmax1 = SoftMax_Crossentropy()
# Forward pass
conv_output = convolution1.forward(X_sample)
relu_output = relu1.forward(conv_output)
pool_output = maxpool1.forward(relu_output)
flat_output = flatten1.forward(pool_output)
n_inputs = flat_output.shape[1]
dense1 = Layer_Dense(n_inputs, n_neurons=num_classes)
dense_output = dense1.forward(flat_output)
softmax_Loss = softmax1.forward(dense_output, Y_sample)
print("Sample Loss:", softmax_Loss)
# backward pass
dvalues = softmax1.backward(softmax1.output, Y_sample)
d_dense = dense1.backward(dvalues)
d_flat = flatten1.backward(d_dense)
d_pool = maxpool1.backward(d_flat)
d_relu = relu1.backward(d_pool)
d_conv = convolution1.backward(d_relu)
print("Grad Shape Conv:", d_conv.shape)
print("Grad Shape dense:", d_dense.shape)

