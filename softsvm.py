import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt

# load question 2 data
data = np.load('EX2q2_mnist.npz')
trainX = data['Xtrain']
testX = data['Xtest']
trainy = data['Ytrain']
testy = data['Ytest']

trainy = np.reshape(trainy, (len(trainy), 1))
testy = np.reshape(testy, (len(testy), 1))


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """

    H = create_H(len(trainX), len(trainX[0]), l)
    # print("H size is:")
    # print(H.size)
    A = create_A(len(trainX), len(trainX[0]), trainX, trainy)
    # print("A size is:")
    # print(A.size)
    v = create_v(len(trainX))
    # print("v size is:")
    # print(v.size)
    u = create_u(len(trainX), len(trainX[0]))
    # print("u size is:")
    # print(u.size)
    sol = solvers.qp(H, u, -A, -v)
    print(sol)
    # print(type(sol['x']))
    w = np.array(sol['x'][:len(trainX[0])])
    # print(type(w[0]))
    # print(w[0].dtype)
    # w = w.astype('float64')
    for i in range(len(w)):
        # print(w[i][0])
        w[i] = w[i][0]
    # print(ret[:len(trainX[0])].shape)
    # print(ret[:len(trainX[0])].shape)
    return w  # [:len(trainX[0])]


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


def simple_test():
    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    countgood = 0
    for i in range(100):
        if (np.sign(testX[i] @ w) == testy[i]):
            countgood += 1

    print("simple test: ", countgood)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")

    # print(np.linalg.norm(w))


def basic_checks():
    A = matrix([0, 1, 2, 3], (2, 2))
    print(A.size[0])


def create_H(m, d, l):
    a = np.identity(d)  # d x d
    b = np.zeros((d, m))  # d x m

    c = np.zeros((m, d))  # m x d
    d = np.zeros((m, m))  # m x m

    H = np.block([[a, b],
                  [c, d]])
    H = H * 2 * l
    H = matrix(H)
    return H


def create_A(m, d, X_train, y_train):
    a = np.zeros((m, d))  # m x d
    b = np.identity(m)  # m x m
    for i in range(len(X_train)):
        X_train[i] = X_train[i] * y_train[i]
    A = np.block([[a, b],
                  [X_train, b]])
    A = matrix(A)
    return A


def create_v(m):
    a = np.zeros(m)  # m x 1
    b = np.ones(m)  # m x 1

    v = np.block([[a, b]]).T
    v = matrix(v)
    return v


def create_u(m, d):
    a = np.zeros(d)  # d x 1
    b = (1 / m) * np.ones(m)  # m x 1

    u = np.block([[a, b]]).T
    u = matrix(u)
    return u


def predict_svm(w, x_test):
    ar = np.asarray([np.sign(sample @ w) for sample in x_test])
    predict = np.reshape(ar, (ar.shape[0], 1))
    return predict


def first_expirement_new():
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    lambdas = [10 ** n for n in range(1, 11)]
    avg_train_error, min_train_error, max_train_error, train_error = [], [], [], []

    for lamda in lambdas:
        test_means, train_means, ws = [], [], []
        for i in range(10):
            x_train, y_train = gensmallm([trainX], [trainy], 100)
            ws.append(softsvm(lamda, x_train, y_train))

            for w in ws:
                pred = predict_svm(w, testX)
                pred = np.reshape(pred, (pred.shape[0],))
                train_pred = predict_svm(w, trainX)
                train_pred = np.reshape(train_pred, (train_pred.shape[0],))
                test_means.append(np.mean(testy != pred))
                train_means.append(np.mean(trainy != train_pred))

        avg_train_error.append(np.mean(test_means))
        train_error.append(np.mean(train_means))
        min_train_error.append(min(test_means))
        max_train_error.append(max(test_means))
    print(train_error)
    #####################3 Important #######################33
    train_error_1000, avg_error_1000 = second_expirement_new()
    plot_new(avg_train_error, train_error, min_train_error, max_train_error, avg_error_1000, train_error_1000)


def plot_new(avg_error, train_error, min_error, max_error, avg_error_1000, train_error_1000):
    x = np.arange(1, 11)
    x_b = [1, 3, 5, 8]
    plt.plot(x, avg_error, label="test Error", color="purple")
    plt.plot(x, train_error, label="train Error", color="red")
    plt.scatter(x_b, avg_error_1000, s=30, label="large sample test error", color="blue", zorder=2)
    plt.scatter(x_b, train_error_1000, s=30, label="large sample train error", color="green", zorder=2)
    # plt.bar(x, min_error, label="Min Error")
    # plt.bar(x, max_error, label="Max Error", bottom=min_error)
    train_bars = np.zeros((10))
    test_bars = np.zeros((10))
    for i in range(10):
        train_bars[i] = max_error[i] - min_error[i]
    print(train_bars)
    plt.errorbar(x, avg_error, yerr=train_bars, color="purple")
    plt.title("softsvm - Q2.b")
    plt.xlabel("lambda to the power of")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


def first_expirement():
    m = 100
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    expirement_result_test = np.zeros((10, 10))
    expirement_result_train = np.zeros((10, 10))
    # Get a random m training examples from the training set
    for i in range(10):
        # _trainX, _trainy = gensmallm([trainX], [trainy[0]], 100)
        indices = np.random.permutation(trainX.shape[0])
        _trainX = trainX[indices[:m]]
        _trainy = trainy[indices[:m]]
        for j in range(10):
            expirement_l = 10 ** (j + 1)
            w = softsvm(expirement_l, _trainX, _trainy)
            # pred = predict_svm(w, testX)
            # pred = np.reshape(pred, (pred.shape[0],))
            # train_pred = predict_svm(w, trainX)
            # train_pred = np.reshape(train_pred, (train_pred.shape[0],))
            # expirement_result_train[i][j] = np.mean(trainy != pred)
            # expirement_result_test[i][j] = np.mean(testy != train_pred)
            # predict = predict_svm(w, testX)
            # expirement_result_test[i][j] = np.mean(testy != predict)
            countgood_train = 0
            countgood_test = 0
            for k in range(len(testX)):
                if (np.sign(testX[k] @ w) == testy[k]):
                    countgood_test += 1

            for ind in range(len(_trainX)):
                # print(type(_trainX[ind]))
                # print(type(w[0]))
                if (np.sign(_trainX[ind] @ w) == _trainy[ind]):
                    countgood_train += 1

            countgood_test /= len(testX)
            expirement_result_test[i][j] = countgood_test
            countgood_train /= len(_trainX)
            expirement_result_train[i][j] = countgood_train
    print("train")
    print(expirement_result_train)
    print("test")
    print(expirement_result_test)

    avg_train = np.mean(expirement_result_train, axis=0)
    avg_test = np.mean(expirement_result_test, axis=0)

    min_train = np.min(expirement_result_train, axis=0)
    min_test = np.min(expirement_result_test, axis=0)

    max_train = np.max(expirement_result_train, axis=0)
    max_test = np.max(expirement_result_test, axis=0)
    print(avg_train, min_train, max_train)
    print(avg_test, min_test, max_test)

    plot_first_exp(avg_train, min_train, max_train, avg_test, min_test, max_test)


def plot_first_exp(train_avg, train_mini, train_maxi, test_avg, test_mini, test_maxi):
    x = np.arange(1, 11)
    train_bars = np.zeros((10))
    test_bars = np.zeros((10))
    plt.xlabel("10 to the power of n ")
    plt.ylabel("error")
    plt.plot(x, 1 - train_avg, label="train error")
    plt.plot(x, 1 - test_avg, label="test error")
    # for i in range(10):
    #     bars[i] = maxi[i] - mini[i]
    # print(bars)
    # plt.errorbar(x, avg, yerr=bars)
    plt.legend()
    plt.show()


def second_expirement():
    m = 1000
    ns = [1, 3, 5, 8]
    expirement_result_test = np.zeros((4))
    expirement_result_train = np.zeros((4))
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]
    for i in range(len(ns)):
        w = softsvm(10 ** ns[i], _trainX, _trainy)
        countgood_test = 0
        countgood_train = 0
        for k in range(len(testX)):
            if (np.sign(testX[k] @ w) == testy[k]):
                countgood_test += 1
        expirement_result_test[i] = countgood_test
        expirement_result_test[i] /= len(testX)
        for k in range(len(_trainX)):
            if (np.sign(_trainX[k] @ w) == _trainy[k]):
                countgood_train += 1
        expirement_result_train[i] = countgood_train
        expirement_result_train[i] /= len(_trainX)

    print(expirement_result_test)

    return expirement_result_train, expirement_result_test


def second_expirement_new():
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    lambdas = [1, 3, 5, 8]
    avg_train_error, min_train_error, max_train_error, train_error = [], [], [], []

    for lamda in lambdas:
        test_means, train_means, ws = [], [], []
        x_train, y_train = gensmallm([trainX], [trainy], 1000)
        ws.append(softsvm(lamda, x_train, y_train))
        for w in ws:
            pred = predict_svm(w, testX)
            pred = np.reshape(pred, (pred.shape[0],))
            train_pred = predict_svm(w, trainX)
            train_pred = np.reshape(train_pred, (train_pred.shape[0],))
            test_means.append(np.mean(testy != pred))
            train_means.append(np.mean(trainy != train_pred))

        avg_train_error.append(np.mean(test_means))
        train_error.append(np.mean(train_means))
        min_train_error.append(min(test_means))
        max_train_error.append(max(test_means))

    # plot_new_sec(avg_train_error, train_error, min_train_error, max_train_error)
    return train_error, avg_train_error


def plot_new_sec(avg_error, train_error, min_error, max_error):
    x = [1, 3, 5, 8]  # np.arange(1,3,5,8)

    # train_bars = np.zeros((4))
    plt.bar(x, min_error, label="Min Error")
    plt.bar(x, max_error, label="Max Error", bottom=min_error)
    plt.scatter(x, avg_error, s=200, label="test Error", color="blue", zorder=2)
    plt.scatter(x, train_error, s=200, label="train Error", color="yellow", zorder=2)
    # for i in range(4):
    #     train_bars[i] = max_error[i] - min_error[i]
    # print(train_bars)
    # plt.errorbar(x, avg_error, yerr=train_bars, color="purple")
    plt.title("Second expirement")
    plt.xlabel("power of lambda")
    plt.ylabel("Error")
    plt.xticks(x)
    plt.legend()
    plt.show()
    # plt.plot(x, avg_error, label="test Error", color="purple")
    # plt.plot(x, train_error, label="train Error", color="red")
    # # plt.bar(x, min_error, label="Min Error")
    # # plt.bar(x, max_error, label="Max Error", bottom=min_error)
    # train_bars = np.zeros((4))
    # for i in range(4):
    #     train_bars[i] = max_error[i] - min_error[i]
    # print(train_bars)
    # plt.errorbar(x, avg_error, yerr=train_bars, color="purple")
    # plt.title("softsvm")
    # plt.xlabel("lambda to the power of")
    # plt.ylabel("Error")
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # basic_checks()

    # softsvm()
    simple_test()
    # first_expirement()
    # second_expirement()
    first_expirement_new()
    second_expirement_new()
