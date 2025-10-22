import numpy as np
class ReLu_Activation:
    def __init__(self):
        self.output = None
        self.dvalues = None
        self.dinputs = None
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    

    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs