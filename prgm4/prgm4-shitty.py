from random import seed
import pandas as pd
import numpy as np


def split_dataset(dataset, train_perc=0.8):
    # np.random.shuffle(dataset)
    data_len = len(dataset)
    # print('length of dataset:' + str(data_len))
    train_index = int(data_len * train_perc)

    train = dataset[:train_index, :]
    test = dataset[train_index:, :]
    return train, test


# np.exp calculate e to the power of input array
def sigmoid(activation):
    return 1.0 / (1.0 + np.exp(-activation))


def compute_loss(prediction, actual):
    # return -sum(actual*log(prediction))
    # print(actual.T-prediction)
    return 0.5 * np.sum((actual.T - prediction) * (actual.T - prediction))


def back_prop(train_X, W1, W2, layer1_output, layer2_output, actual_output):
    # find error in output unit
    difference = actual_output.T - layer2_output
    delta_output = layer2_output * (1 - layer2_output) * difference
    delta_hidden = layer1_output * (1 - layer1_output) * W2.T.dot(delta_output)
    deltaW2 = lr * (delta_output.dot(layer1_output.T) / n_train)
    deltaW1 = lr * (delta_hidden.dot(train_X) / n_train)

    return (deltaW1, deltaW2)


def train_network(train_X, train_y):
    n_input = train_X.shape[1]  # the number of columns in the training data
    W1 = np.random.random((n_hidden, n_input))
    # print(W1)
    W2 = np.random.random((num_classes, n_hidden))

    for epoch in range(n_epoch):
        layer1_output = sigmoid(W1.dot(train_X.T))
        # print(W1.dot(train_X.T))
        # print(layer1_output)
        layer2_output = sigmoid(W2.dot(layer1_output))

        (deltaW1, deltaW2) = back_prop(train_X, W1, W2, layer1_output,
                                       layer2_output, train_y)

        W2 = W2 + deltaW2
        W1 = W1 + deltaW1

        if epoch % 100 == 0:
            loss = compute_loss(layer2_output, train_y)
            print(str.format('loss in {0}th epoch is {1}', epoch, loss))

    return (W1, W2)


def evaluate(test_X, test_y, params):
    (W1, W2) = params
    layer1_output = sigmoid(W1.dot(test_X.T))
    final = sigmoid(W2.dot(layer1_output))
    print(final)
    prediction = final.argmax(axis=0)
    print(prediction)
    print(test_y)
    return np.sum(prediction == test_y) / len(test_y)


def convert_to_OH(data, num_classes):
    # create an array to store the one hot vectors
    one_hot = np.zeros((len(data), num_classes))
    # print('data', data)
    one_hot[np.arange(len(data)), data - 1] = 1
    # print(one_hot)
    return one_hot

    # load and perpare data


filename = 'dataset.csv'
df = pd.read_csv(filename, dtype=np.float64, header=None)
dataset = np.array(df)
(train, test) = split_dataset(dataset)
n_train = len(train)
print("n_train : " + str(n_train))
n_test = len(test)
print("n_test : " + str(n_test))
# evaluate algorithm
lr = 0.2
n_epoch = 3000
# determine the number of classes
# np.unique returns an array of unique elementsin input array
num_classes = len(np.unique(dataset[:, -1]))
# print("No. of Classes : ", str(num_classes))
train_one_hot = convert_to_OH(train[:, -1].astype(int), num_classes)

n_hidden = 20
params = train_network(train[:, :-1], train_one_hot)
accuracy = evaluate(test[:, :-1], test[:, -1], params) * 100
print("Mean Accuracy : %.3f%%" % accuracy)
