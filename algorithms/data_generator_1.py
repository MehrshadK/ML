import numpy as np


def DataGenerator_1():
    """sample size is data set size. you can change this for more test"""
    sample_number = 500

    """ generate data for class1 with mean_1 and cov1 """
    mean_0 = [0, 0]
    cov0 = np.array([[0.5, 3.], [3., 0.1]])
    data_0 = np.random.multivariate_normal(mean_0, cov0, sample_number, check_valid='ignore')

    """ generate data for class_2 with mean_2 and cov_2 """
    mean_1 = [1, 2]
    cov_1 = np.array([[0.25, 0.3], [0.3, 1.]])
    data_1 = np.random.multivariate_normal(mean_1, cov_1, sample_number, check_valid='ignore')

    """data to merge two data and add lable 1 and 2 to each class."""
    data = np.zeros((2 * sample_number, 3))
    for i in range(0, 50):
        data[i, 0] = data_0[i, 0]
        data[i, 1] = data_0[i, 1]
        data[i, 2] = 0
    i = 0

    for j in range(50, 100):
        data[j, 0] = data_1[i, 0]
        data[j, 1] = data_1[i, 1]
        data[j, 2] = 1
        i += 1

    np.random.shuffle(data)
    np.savetxt("data_2.txt", data)
    return data


if __name__ == '__main__':
    DataGenerator_1()
