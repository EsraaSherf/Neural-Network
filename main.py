
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from random import randrange




class NeuralNetwork:
    def __init__(self, X, y, w1, w2):
        self.input = X
        self.weights1 = w1
        self.weights2 = w2
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        # calculate hidden layer layer
        self.input = np.hstack((np.matrix(np.ones(self.input.shape[0])).T, self.input))
        self.layer1 = sigmoid(np.dot(self.input, self.weights1.T))
        # calculate output
        self.layer1 = np.hstack((np.matrix(np.ones(self.layer1.shape[0])).T, self.layer1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2.T))
        self.y_predict = np.array([np.argmax(v) + 1 for v in self.output]).reshape(self.y.shape)

    # Calculate Accuracy
    def Accuracy(self):
        it = 0
        for i in range(len(self.y)):
            if self.y[i] == self.y_predict[i]:
                it+= 1
        result = (it/ len(self.y)) * 100
        return result

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

# Display sample image randomly.
def displySample(X):
    img = X[randrange(X.shape[0])]
    img = img.reshape(20, 20)

    plt.figure(figsize=(0.5, 0.5))
    plt.imshow(img, cmap='gray')
    plt.show()


# 1- Load dataset
dataset = loadmat('data.mat')
X = dataset['X']
y = dataset['y']

print("number of samples = ", X.shape[0])
print("Number of fetures = ", X.shape[1], " representing ", np.sqrt(X.shape[1]), 'x', np.sqrt(X.shape[1]), ' image')
displySample(X)
# =======================================
weights = loadmat('weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']
print("Theta1 shape = ", Theta1.shape)
print("Theta2 shape = ", Theta2.shape)

nn = NeuralNetwork(X, y, Theta1, Theta2)
nn.feedforward()
print("Accuracy = ", nn.Accuracy(), )






