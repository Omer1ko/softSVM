import math
import softsvm
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmbf(l: float, sigma: float, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m = len(trainX)

    G = create_G(trainX, sigma)

    H = create_H(G, l)

    u = create_u(m)

    v = create_v(m)

    A = create_A(G, trainy)

    sol = solvers.qp(H, u, -A, -v)
    # print(sol)
    # print(type(sol['x']))
    ret = np.array(sol['x'])
    # print(ret)
    # print(ret.shape)
    # print(ret[:len(trainX)].shape)
    # print(ret[:len(trainX)])
    return ret[:len(trainX)]

    # raise NotImplementedError()


def simple_test():
    # load question 4 data
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvmbf(10, 0.5, _trainX, _trainy)

    countgood = 0
    # print(len(testX))
    for i in range(len(testX)):
        if (cluster_x(w, _trainX, testX[i], 0.1) == testy[i]):
            countgood += 1

    # print("good assignment: ", countgood / len(testX))

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


def create_G(trainX, sigma):
    G = np.zeros((len(trainX), len(trainX)))

    for i in range(len(trainX)):
        # print(i)
        for j in range(len(trainX)):
            temp = (np.linalg.norm(trainX[i] - trainX[j])) ** 2
            temp = -temp / (2 * sigma)
            G[i][j] = math.e ** temp
    # print(G)
    G = matrix(G)
    return G


def create_H(G, l):
    # G is of size m x m
    G = np.array(G)
    m = len(G)
    # print(m)
    a = np.zeros((m, m))

    H = np.block([[G, a],
                  [a, a]])
    H = 2 * l * H
    # print(np.linalg.eigvals(H))
    # print(H.shape[0])
    H = H + np.identity(H.shape[0]) * 0.000000001

    # print(min(np.linalg.eigvals(H)))
    H = matrix(H)

    return H


def create_u(m):
    a = np.zeros((m))
    b = (1 / m) * np.ones(m)  # m x 1

    u = np.block([[a, b]]).T
    u = matrix(u)
    return u


def create_v(m):
    a = np.zeros((m))
    b = np.ones((m))  # m x 1

    v = np.block([[a, b]]).T
    v = matrix(v)
    return v


def create_A(G, trainY):
    G = np.array(G)
    m = len(G)
    a = np.zeros((m, m))  # m x m
    b = np.identity(m)  # m x m

    for i in range(m):
        G[i] = G[i] * trainY[i]

    A = np.block([[a, b],
                  [G, b]])

    A = matrix(A)
    return A


def cluster_x(a, trainX, x, sigma):
    m = len(trainX)

    sumi = 0
    # print(m)
    for i in range(m):
        sumi += a[i] * math.e ** (-((np.linalg.norm(trainX[i] - x)) ** 2 / 2 * sigma))
    if (sumi > 0):
        return 1
    return -1


def a():
    # load question 4 data
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    for i in range(len(trainX)):
        if trainy[i] == 1:
            plt.scatter(trainX[i][0], trainX[i][1], c="blue")
        else:
            plt.scatter(trainX[i][0], trainX[i][1], c="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Training data points")
    plt.show()


def b():
    # load question 4 data
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    # lamdas = np.array([1, 10, 100])  # np.array([10])#
    # sigmas = np.array([0.01, 0.5, 1])  # np.array([0.5])  #
    # num_of_samples = len(trainX)
    # best_count_good, best_sigma, best_lamda = 0, 0, 0
    # all_errors = []
    # for lamda in lamdas:
    #     for sigma in sigmas:
    #         print(f"RBF Soft SVM with:lamda={lamda}, sigma={sigma}")
    #         alpha = softsvmbf(lamda, sigma, trainX, trainy)
    #         countgood = 0
    #         for i in range(len(testy)):  # range(100):
    #             if (cluster_x(alpha, trainX, trainX[i], sigma) == testy[i]):
    #                 countgood += 1
    #             if countgood > best_count_good:
    #                 best_count_good = countgood
    #                 best_sigma = sigma
    #                 best_lamda = lamda
    #                 err = 1 - (countgood / len(testy))
    #         all_errors.append(1 - (countgood / len(testy)))
    # print(all_errors)
    # print(f"The best sigma is {best_sigma} and the best lamda is {best_lamda} ")
    sigma = best_sigma = 1
    lamda = 100
    alpha = softsvmbf(lamda, sigma, trainX, trainy)
    # plot_b(trainX, alpha, best_sigma, title="Training data points classified")
    plot_b_4343(trainX, alpha, "real")
    #
    # print(f"The test error is {err}")
    # non - kernel, linear code is here:
    #best_lamda = 100
    # plot_b_q1(best_lamda, trainX, trainy, "Svm from Q1")


def plot_b(trainX, alpha, sigma, title):
    for i in range(len(trainX)):
        if cluster_x(alpha, trainX, trainX[i], sigma) == 1:
            plt.scatter(trainX[i][0], trainX[i][1], c="blue")
        else:
            plt.scatter(trainX[i][0], trainX[i][1], c="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


def plot_b_4343(trainX, alpha, title):
    sigmas = [0.01, 0.5, 1]
    customTrain = np.arange(start=-7, stop=7, step=0.2)
    customTrain = np.transpose([np.tile(customTrain, len(customTrain)), np.repeat(customTrain, len(customTrain))])
    print(type(customTrain))
    for sigma in sigmas:
        for i in range(len(customTrain)):
            if i % 1000 == 0:
                print(i)
            if cluster_x(alpha, trainX, customTrain[i], sigma) == 1:
                plt.scatter(customTrain[i][0], customTrain[i][1], c="blue")
            else:
                plt.scatter(customTrain[i][0], customTrain[i][1], c="red")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"lambda - {100}, sigma - {sigma}")
        plt.show()


def plot_b_q1(best_lamda, trainX, trainy, title):
    w = softsvm.softsvm(best_lamda, trainX, trainy)
    train_pred = softsvm.predict_svm(w, trainX)
    train_pred = np.reshape(train_pred, (train_pred.shape[0],))
    countbad = 0
    for i in range(len((train_pred))):
        if train_pred[i] != trainy[i]:
            countbad += 1
    print(f"The error is {countbad / len(train_pred)}")
    for i in range(len(trainX)):
        if train_pred[i] == 1:
            plt.scatter(trainX[i][0], trainX[i][1], c="blue")
        else:
            plt.scatter(trainX[i][0], trainX[i][1], c="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


def d():
    # load question 4 data
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    lamda = 100
    sigmas = np.array([0.01, 0.5, 1])
    for sigma in sigmas:
        alpha = softsvmbf(lamda, sigma, trainX, trainy)
        for i in range(len(trainX)):
            if (cluster_x(alpha, trainX, trainX[i], sigma) == 1):  # testy[i]):
                plt.scatter(trainX[i][0], trainX[i][1], c="blue")
            else:
                plt.scatter(trainX[i][0], trainX[i][1], c="red")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"lamda - 100, sigma - {sigma}")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    print('Running Simple Test')
    simple_test()
    print('Running Q 2.a')
    a()
    print('Running Q 2.b, it will take a long time, coffee time')
    b()
    print('Running Q 2.d')
    d()


