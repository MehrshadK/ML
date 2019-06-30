# kNN implemented in Python
import math
import operator
import data_generator_1
import matplotlib.pyplot as plt


def load_dataset(dataset, splt):
    return dataset[:splt], dataset[splt:]


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclidean_distance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(predictions)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    # prepare data
    split = 0.6

    dataset_1 = data_generator_1.DataGenerator_1().tolist()
    n_samples = len(dataset_1)
    splt = int(split * n_samples)
    trainingSet, testSet = load_dataset(dataset_1, splt)

    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))

    knn = []
    k_knn = [i for i in range(1, 21, 2)]
    # generate predictions
    for k in k_knn:
        predictions = []
        for x in range(len(testSet)):
            neighbors = get_neighbors(trainingSet, testSet[x], k)
            result = get_response(neighbors)
            predictions.append(result)
            # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
        accuracy = get_accuracy(testSet, predictions)
        knn.append(accuracy)
        print('Accuracy: ' + repr(accuracy) + '%')

    x_label = "number of k in knn"
    y_label = "accuracy with split="
    y_label += str(split)

    plt.plot(k_knn, knn)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


if __name__ == '__main__':
    main()
