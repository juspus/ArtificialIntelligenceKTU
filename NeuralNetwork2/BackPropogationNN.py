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


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1-fx)

def mse_loss(y_true, y_preds):
    return ((y_true - y_preds)**2).mean()
class OurNN:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

        bias = 0
        
    def feedForward(self, x):
        h1 = sigmoid(self.w1 + x[0]+ self.w2 + x[1] + self.b1)
        h2 = sigmoid(self.w3 + x[0]+ self.w4 + x[1] + self.b2)
        o1 = sigmoid(self.w5 + h1+ self.w6 + h2 + self.b3)
        
        return o1

    def train(self, data, target_vector):

        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, target_vector):
                h1 = self.w1 + x[0]+ self.w2 + x[1] + self.b1
                out_h1 = sigmoid(h1)
                h2 = self.w3 + x[0]+ self.w4 + x[1] + self.b2
                out_h2 = sigmoid(h2)
                o1 = self.w5 + h1+ self.w6 + h2 + self.b3
                out_o1 = sigmoid(o1)
                y_pred = out_o1

                d_L_d_ypred = -2*(y_true-y_pred)

                d_ypred_d_w5 =  h1 * deriv_sigmoid(o1)
                d_ypred_d_w6 =  h2 * deriv_sigmoid(o1)
                d_ypred_d_b3 =  deriv_sigmoid(o1)

                d_ypred_d_h1 =  self.w5 * deriv_sigmoid(o1)
                d_ypred_d_h2 =  self.w6 * deriv_sigmoid(o1)

                d_h1_d_w1 = x[0] * deriv_sigmoid(h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(h1)
                d_h2_d_b2 =  deriv_sigmoid(h2)

                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_b2

                self.w5 -= learn_rate * d_L_d_ypred * d_h1_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_h1_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_h1_d_b3

                if epoch % 10 ==0:
                    y_preds = np.apply_along_axis(self.feedForward, 1, data)
                    loss = mse_loss(target_vector, y_pred)
                    print("Epoch %d loss: %.3f"  % (epoch, loss))
                    
data = np.array([[-2, -1],
                [25, 6],
                [17, 4],
                [-15, -6]])

target_vector = ([1, 0, 0 ,1,])
tinklas = OurNN()
tinklas. train(data, target_vector)