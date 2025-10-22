import numpy as np
class MaxPool:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.output = None
        self.dvalues = None
        
    def forward(self, inputs):
        self.inputs = inputs
        self.batch_size, self.input_depth, self.input_height, self.input_width = inputs.shape
        self.output_height = (self.input_height - self.pool_size) // self.stride + 1
        self.output_width = (self.input_width - self.pool_size) // self.stride + 1
        self.output = np.zeros((self.batch_size, self.input_depth, self.output_height, self.output_width))
        
        
        for i in range(self.output_height):
            for j in range(self.output_width):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.pool_size
                w_end = w_start + self.pool_size  
                input_slice = inputs[:, :, h_start:h_end, w_start:w_end]
                self.output[:, :, i, j] = np.max(input_slice, axis=(2, 3))
        return self.output
                
    def backward(self, dvalues):
        self.dinputs = np.zeros_like(self.inputs)
        
        for i in range(self.output_height):
            for j in range(self.output_width):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.pool_size
                w_end = w_start + self.pool_size  
                input_slice = self.inputs[:, :, h_start:h_end, w_start:w_end]
                max_values = np.max(input_slice, axis=(2, 3), keepdims=True)
                mask = (input_slice == max_values)
                self.dinputs[:, :, h_start:h_end, w_start:w_end] += mask * (dvalues[:, :, i, j])[:, :, None, None]
        return self.dinputs
    