import numpy as np
class Convolution:
    def __init__(self, num_filters, filter_size, input_depth, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_depth = input_depth
        self.stride = stride
        self.padding = padding
        self.filter = np.random.randn(num_filters, input_depth, filter_size, filter_size) * 0.1
        self.biases = np.zeros((num_filters, 1))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.batch_size, self.input_depth, self.input_height, self.input_width = inputs.shape
        input_padded = np.pad(inputs, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        self.input_padded = input_padded
        self.output_height = (self.input_height - self.filter_size + 2 * self.padding)// self.stride + 1
        self.output_width = (self.input_width - self.filter_size + 2 * self.padding)// self.stride + 1
        self.output = np.zeros((self.batch_size, self.num_filters, self.output_height, self.output_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.filter_size
                w_end = w_start + self.filter_size  
                input_slice = input_padded[:, :, h_start:h_end, w_start:w_end]
                for f in range(self.num_filters):
                    conv_sum = np.sum(input_slice * self.filter[f, :, :, :], axis=(1, 2, 3))
                    self.output[:, f, i, j] = conv_sum + self.biases[f]
        return self.output
    def backward(self, dvalues):
        self.dinputs = np.zeros_like(self.input_padded)
        self.dfilters = np.zeros_like(self.filter)
        self.dbiases = np.zeros_like(self.biases)
        self.dbiases = np.sum(dvalues, axis=(0, 2, 3)).reshape(self.num_filters, -1)
        for i in range(self.output_height):
            for j in range(self.output_width):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.filter_size
                w_end = w_start + self.filter_size  
                input_slice = self.input_padded[:, :, h_start:h_end, w_start:w_end]
                for f in range(self.num_filters):
                    self.dfilters[f] += np.sum(input_slice * (dvalues[:, f, i, j])[:, None, None, None], axis=0)
                for n in range(self.batch_size):
                    self.dinputs[n, :, h_start:h_end, w_start:w_end] += np.sum((self.filter[:, :, :, :] * (dvalues[n, :, i, j])[:, None, None, None]), axis=0)   
                    
        if self.padding != 0:
            self.dinputs = self.dinputs[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return self.dinputs                 
                
                
        
        