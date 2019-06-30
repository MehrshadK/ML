from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import copy
from data_generator_1 import DataGenerator_1

data = DataGenerator_1()
""" x is data and y is target """
x = data[:, :-1]
y = data[:, -1]
y = np.asarray((y), dtype=np.int32)

print("##########################################################################################")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print('Train set: ', len(x_train))
print('Test set: ', len(x_test))
train_labels = np.zeros((len(y_train), 2))
test_labels = np.zeros((len(y_test), 2))

for i in range(len(y_train)):
    train_labels[i, [y_train[i]]] = 1

for i in range(len(y_test)):
    test_labels[i, [y_test[i]]] = 1

learning_rate = 0.2


class neural_network:
    def __init__(self, n_inputs, n_hidden, n_targets, lr):
        self.input_size = n_inputs
        self.hidden_size = n_hidden
        self.target_size = n_targets
        self.learning_rate = lr
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.random.randn(n_hidden)
        self.weights2 = np.random.randn(self.hidden_size, self.target_size)
        self.bias2 = np.random.randn(n_targets)
        self.correct_count = 0

    def feedforward(self, X):
        self.layer1 = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        return self.output

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_prime(self, s):
        return s * (1.0 - s)

    def backprop(self, X, y):
        self.y = y
        self.output_error = y - self.output
        self.output_delta = self.output_error * self.sigmoid_prime(self.output)

        self.layer1_error = self.output_delta.dot(self.weights2.T)
        self.layer1_delta = self.layer1_error * self.sigmoid_prime(self.layer1)

        try:
            d_weights2 = self.output_delta * self.layer1 * self.learning_rate
        except ValueError:
            row = len(self.layer1)
            d_weights2 = self.output_delta * self.layer1.reshape(row, 1) * self.learning_rate

        d_bias2 = self.output_delta * self.learning_rate
        shape_X = X.shape[0]
        d_weights1 = X.reshape(shape_X, 1) * self.layer1_delta * self.learning_rate
        d_bias1 = self.layer1_delta * self.learning_rate

        # update the weights with the derivative (slope) of the loss function
        # row, column = self.weights2.shape
        self.weights1 += d_weights1
        self.bias1 += d_bias1
        self.weights2 += d_weights2
        self.bias2 += d_bias2

    def mean_squared_error(self):
        return sum(np.power((self.y - self.output), 2)) / 2

    def train(self, X, y):
        o = self.feedforward(X)
        self.backprop(X, y)

    def predict(self, X, y, yt):
        self.out = copy.deepcopy(self.feedforward(X))
        max = self.out.max()
        for i in range(len(self.out)):
            if self.out[i] == max:
                self.out[i] = 1
            else:
                self.out[i] = 0
        # print("Predicted data based on trained weights: ")
        # print("Predicted output is > ", self.out, " , ", "Actual output is > ", yt)
        if np.array_equal(self.out, y):
            self.correct_count += 1

    def accuracy(self):
        print("----------------------------------------------------------")
        print("correct > ", self.correct_count, "total > ", len(x_test))
        print("accuracy > ", (self.correct_count / len(x_test)) * 100)


def main():
    total = 0
    mse = []
    nn = neural_network(x_train.shape[1], 8, 2, learning_rate)
    while True:
        for i in range(len(x_train)):
            nn.train(x_train[i], train_labels[i])
        mse.append(nn.mean_squared_error())
        total += 1

        if nn.mean_squared_error() < 0.000001:
            break

        if total % 1000 == 0:
            print("number of epochs now > ", total)
        if total == 15000:
            break

    for i in range(len((x_test))):
        nn.predict(x_test[i], test_labels[i], y_test[i])

    print("total number of epochs > ", total)
    nn.accuracy()

    epochs = [i for i in range(total)]
    plt.plot(epochs, mse)
    plt.xlabel("number of epochs")
    plt.ylabel("MSE")
    plt.show()


if __name__ == '__main__':
    main()
