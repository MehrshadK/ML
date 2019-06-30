from sklearn.model_selection import train_test_split
import numpy as np
# from numpy.linalg import inv
import data_generator_1
import matplotlib.pyplot as plt

""" x is data and y is target """
data = data_generator_1.DataGenerator_1()
x = data[:, :-1]
y = data[:, -1]

print("##########################################################################################")
bayes_accuracy = []
ls = [0.3, 0.6, 0.8]
for i in ls:
    split_ratio = i
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - split_ratio)
    print('Train set: ', len(x_train))
    print('Test set: ', len(x_test))

    x_train_labeled = np.column_stack((x_train, y_train))

    x_train_0, x_train_1 = [], []

    """ seprate train data with class lebel"""
    for i in x_train_labeled:
        if i[-1] == 0:
            x_train_0.append(i[:-1])
        else:
            x_train_1.append(i[:-1])

    x_train_0 = np.asarray(x_train_0)
    x_train_1 = np.asarray(x_train_1)

    """ calculate mean for train dataset """


    def mean(numbers):
        train_mean = np.mean(numbers, axis=0)
        return train_mean


    """ assing mean matrix for each class dataset """
    mean_data_0 = mean(x_train_0)
    mean_data_1 = mean(x_train_1)

    """ calculate covariance for train dataset """


    def covariance(numbers):
        train_cov = np.cov(numbers.T)
        return train_cov


    """ assing covariance matrix for each class dataset """
    cov_data_0 = covariance(x_train_0)
    cov_data_1 = covariance(x_train_1)

    """ calculate gi(x) """


    def calculate_prob(x, mean_matrix, cov_matrix):
        part_1 = -(np.log(2 * np.pi))
        part_2 = (np.log(np.linalg.det(cov_matrix)))
        part_3 = np.matmul(np.matmul(np.transpose(x - mean_matrix), (np.linalg.inv(cov_matrix))), (x - mean_matrix))
        answer = part_1 - part_2 - part_3
        return answer


    """ gi(x) for each i """


    def g_i_x():
        g0x, g1x = [], []
        for i in range(len(x_test)):
            g_0_x = calculate_prob(x_test[i, :], mean_data_0, cov_data_0)
            g0x.append(g_0_x)
            g_1_x = calculate_prob(x_test[i, :], mean_data_1, cov_data_0)
            g1x.append(g_1_x)
        return [g0x, g1x]


    """ assign gi(x) to g1(x), g2(x) by call g_i_x() function """
    gi_x_0, gi_x_1 = g_i_x()

    """ predict label of train dataset """


    def predict_label(g0, g1):
        predictions = []
        for i in range(len(g0)):
            if (g0[i] >= g1[i]):
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions


    """ assign predicted label to test_predict function by call predict_lebel() function """
    test_predict = predict_label(gi_x_0, gi_x_1)

    """ calculate accuracy of predicts """


    def accuracy(test_x, test_y):
        count = 0
        for i in range(len(test_x)):
            if test_x[i] == test_y[i]:
                count += 1
        print("accuracy > ", end="")
        return (count / len(test_x)) * 100


    bayes = accuracy(test_predict, y_test)
    print(bayes)
    bayes_accuracy.append(bayes)

plt.plot(['0.3', '0.6', '0.8'], bayes_accuracy)
plt.xlabel("The ratio of training data")
plt.ylabel("accuracy")
plt.show()
