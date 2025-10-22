import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["OMP_NUM_THREADS"] = "8"      # number of CPU threads
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
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
convolution1 = Convolution(num_filters=8, filter_size=3, input_depth=1, stride=1, padding=1)
convolution2 = Convolution(num_filters=16, filter_size=3, input_depth=8, stride=1, padding=1)
relu1 = ReLu_Activation()
relu2 = ReLu_Activation()
maxpool1 = MaxPool(pool_size=2, stride=2)
maxpool2 = MaxPool(pool_size=2, stride=2)
flatten1 = Flatten()
softmax1 = SoftMax_Crossentropy()
learning_rate = 1e-3
epochs = 20
batch_size = 32
dummy_input = np.random.randn(1, 1, 28, 28)
conv_out = convolution1.forward(dummy_input)
relu_out = relu1.forward(conv_out)
pool_out = maxpool1.forward(relu_out)
conv2_out = convolution2.forward(pool_out)
relu2_out = relu2.forward(conv2_out)
pool2_out = maxpool2.forward(relu2_out)
flat_out = flatten1.forward(pool2_out)
n_inputs = flat_out.shape[1]


# Now create dense layer
hidden_dense = Layer_Dense(n_inputs, n_neurons=64)
dense1 = Layer_Dense(64, n_neurons=num_classes)
dense1.weights = dense1.weights.astype(np.float32)
dense1.biases = dense1.biases.astype(np.float32)
hidden_dense.weights = hidden_dense.weights.astype(np.float32)
hidden_dense.biases = hidden_dense.biases.astype(np.float32)
X_train_small = X_train[:10000]  
Y_train_small = Y_train[:10000]

for epoch in range(epochs):
    permutation = np.random.permutation(len(X_train_small))
    X_train_small = X_train_small[permutation]
    Y_train_small = Y_train_small[permutation]
    epoch_Loss = 0.0
    total_epoch_accuracy = 0.0
    for i in range(0, len(X_train_small), batch_size):
        X_batch = X_train_small[i:i+batch_size]
        Y_batch = Y_train_small[i:i+batch_size]
        # Forward pass
        conv_output = convolution1.forward(X_batch)
        #print("Conv output min/max:", conv_output.min(), conv_output.max())
        relu_output = relu1.forward(conv_output)
        #print("ReLU output min/max:", relu_output.min(), relu_output.max())
        pool_output = maxpool1.forward(relu_output)
        conv_2 = convolution2.forward(pool_output)
        relu_output2 = relu2.forward(conv_2)
        pool2 = maxpool2.forward(relu_output2)
        
        #print("Pool output min/max:", pool_output.min(), pool_output.max())
        flat_output = flatten1.forward(pool2)
        #print("Flat output min/max:", flat_output.min(), flat_output.max())
        dense_output_hidden = hidden_dense.forward(flat_output)
        dense_output = dense1.forward(dense_output_hidden)
        #print("Dense output min/max:", dense_output.min(), dense_output.max())
        softmax_Loss = softmax1.forward(dense_output, Y_batch)
        epoch_Loss += softmax_Loss
        predictions = np.argmax(softmax1.output, axis=1)
        accuracy = np.mean(predictions == Y_batch)
        total_epoch_accuracy += accuracy
        # Backward pass
        dvalues = softmax1.backward(softmax1.output, Y_batch)
        d_dense = dense1.backward(dvalues)
        d_hidden_dense = hidden_dense.backward(d_dense)
        d_flat = flatten1.backward(d_hidden_dense)
        d_pool2 = maxpool2.backward(d_flat)
        d_relu2 = relu2.backward(d_pool2)
        d_conv2 = convolution2.backward(d_relu2)
        d_pool = maxpool1.backward(d_conv2)
        d_relu = relu1.backward(d_pool)
        d_conv = convolution1.backward(d_relu)
        # print(f"\n--- Batch {i//batch_size + 1} ---")
        # print(f"Loss for this batch: {epoch_Loss:.4f}")

        # # 1. Check the output of the final layer BEFORE softmax
        # print(f"Dense Output Stats -> Min: {dense_output.min():.6f}, Max: {dense_output.max():.6f}, Mean: {dense_output.mean():.6f}")

        # # 2. Check the gradients calculated for the weights and filters
        # print(f"Dense dWeights Stats -> Min: {dense1.dweights.min():.6f}, Max: {dense1.dweights.max():.6f}, Mean: {dense1.dweights.mean():.6f}")
        # print(f"Conv. dFilters Stats -> Min: {convolution1.dfilters.min():.6f}, Max: {convolution1.dfilters.max():.6f}, Mean: {convolution1.dfilters.mean():.6f}")
        # Update weights and biases
        dense1.weights -= learning_rate * dense1.dweights
        hidden_dense.weights -= learning_rate * hidden_dense.dweights
        dense1.biases -= learning_rate * dense1.dbiases
        hidden_dense.biases -= learning_rate * hidden_dense.dbiases
        convolution1.filter -= learning_rate * convolution1.dfilters
        convolution2.filter -= learning_rate * convolution2.dfilters
        convolution1.biases -= learning_rate * convolution1.dbiases
        convolution2.biases -= learning_rate * convolution2.dbiases
        num_batches = len(X_train_small) / batch_size
    avg_loss = epoch_Loss / num_batches
    avg_accuracy = total_epoch_accuracy / num_batches
    
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.3f}")
print("Training finished. Saving model...")
np.savez('mnist_model.npz', 
         conv1_filters=convolution1.filter,
         conv1_biases=convolution1.biases,
         dense1_weights=dense1.weights,
         dense1_biases=dense1.biases,
         conv2_filters=convolution2.filter,
         conv2_biases=convolution2.biases,
         hidden_dense_weights=hidden_dense.weights,
         hidden_dense_biases=hidden_dense.biases)
print("Model saved successfully as mnist_model.npz")
