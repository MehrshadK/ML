import numpy as np


def DataGenerator_2():
    # sample size is data set size. you can change this for more test
    sample_number = 50

    """ generate data for class_1 with mean_1 and cov_1 """
    mean_0 = [-1.5, 1]
    cov0 = np.array([[0.5, 3.], [3., 1.1]])
    data_0 = np.random.multivariate_normal(mean_0, cov0, sample_number, check_valid='ignore')
    class_zero = np.zeros((sample_number, 1))
    data_0 = np.hstack((data_0, class_zero))

    """ generate data for class_2 with mean_2 and cov_2 """
    mean_1 = [0.5, 0]
    cov_1 = np.array([[0.25, 0.3], [0.3, 1.]])
    data_1 = np.random.multivariate_normal(mean_1, cov_1, sample_number, check_valid='ignore')
    class_one = np.ones((sample_number, 1))
    data_1 = np.hstack((data_1, class_one))

    """ generate data for class_3 with mean_3 and cov_3 """
    mean_2 = [-1.5, -2.25]
    cov2 = np.array([[0.25, 0.3], [0.3, 0.7]])
    data_2 = np.random.multivariate_normal(mean_2, cov2, sample_number, check_valid='ignore')
    class_two = np.full((sample_number, 1), 2)
    data_2 = np.hstack((data_2, class_two))

    data = np.concatenate((data_0, data_1, data_2), axis=0);

    np.random.shuffle(data)
    np.savetxt("data_3.txt", data)
    return data


if __name__ == '__main__':
    DataGenerator_2()
