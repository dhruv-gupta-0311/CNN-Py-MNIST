import numpy as np
class Flatten:
    def __init__(self):
        self.input_shape = None
        self.output = None
        self.dvalues = None
    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.output = inputs.reshape(self.input_shape[0], -1)
        return self.output
    def backward(self, dvalues):
        self.dvalues = dvalues
        self.dinputs = dvalues.reshape(self.input_shape)
        return self.dinputs