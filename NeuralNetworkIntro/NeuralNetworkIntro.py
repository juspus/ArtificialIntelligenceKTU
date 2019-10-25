import numpy as np

def sigmoid (x):
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, input):
        total = 0        
        for i in range(0, len(input)):
            total += self.weights[i]*input[i]
        total = total + self.bias
        
        return sigmoid(total)


class OurNN:
    def __init__(self):
        weight = np.array([0,1,2,3,4])
        bias = 0
        self.h1 = Neuron(weight, bias)
        self.h2 = Neuron(weight, bias)  
        #weight = np.array([0,1])
        self.o1 = Neuron([0,1], bias)
    def feedForward(self, x):
        out_h1 = self.h1.feedForward(x)
        out_h2 = self.h2.feedForward(x)
        out_o1 = self.o1.feedForward([out_h1, out_h2])
        return out_o1
#weights = np.array([0,1,2,3,4])
x = np.array([2,3,4,5,6])
bias = 0
#n = Neuron(weights, bias)
tinklas = OurNN()

print(tinklas.feedForward(x))
#nx = np.array([n.feedForward(x), n.feedForward(x)])
#print(n.feedForward(nx))