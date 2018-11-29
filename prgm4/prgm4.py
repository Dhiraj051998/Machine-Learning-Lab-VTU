import pandas as pd
import numpy as np

n_hidden = 20
learning_rate = 0.1
n_epoch = 30000
train_size = 0


def sigmoid(activation):
    return 1.0 / (1.0 + np.exp(-activation))


def compute_loss(prediction, actual):
    return 0.5 * np.sum((actual.T - prediction) * (actual.T - prediction))


def back_propagation(X_train, w2, layer1_output, layer2_output, actual_output):
    difference = actual_output.T - layer2_output
    delta_output = layer2_output * (1 - layer2_output) * difference
    delta_hidden = layer1_output * (1 - layer1_output) * w2.T.dot(delta_output)
    deltaW2 = learning_rate * (delta_output.dot(layer1_output.T) / train_size)
    deltaW1 = learning_rate * (delta_hidden.dot(X_train) / train_size)

    return deltaW1, deltaW2


def main():
    global train_size
    dataset = pd.read_csv('new_dataset.csv')
    train_size = int(len(dataset) * 0.8)
    X_train = dataset.iloc[:train_size, 1:-1]
    y_train = dataset.iloc[:train_size, -1]

    X_test = dataset.iloc[train_size:, 1:-1]
    y_test = dataset.iloc[train_size:, -1]

    no_of_classes = len(np.unique(dataset.iloc[:, -1]))
    print(no_of_classes)

    one_hot = np.zeros((train_size, no_of_classes))
    one_hot[np.arange(train_size), y_train] = 1

    w1 = np.random.random((n_hidden, 7))

    w2 = np.random.random((no_of_classes, n_hidden))

    for epoch in range(n_epoch):
        layer1_output = sigmoid(w1.dot(X_train.T))

        layer2_output = sigmoid(w2.dot(layer1_output))

        deltaW1, deltaW2 = back_propagation(X_train, w2, layer1_output, layer2_output, one_hot)

        w2 = w2 + deltaW2
        w1 = w1 + deltaW1

        if epoch % 100 == 0:
            loss = compute_loss(layer2_output, one_hot)
            print('loss in {0}th epoch is {1}'.format(epoch, loss))

    layer1_output = sigmoid(w1.dot(X_train.T))
    final = sigmoid(w2.dot(layer1_output))
    print(final)
    prediction = final.argmax(axis=0)
    prediction = np.array(prediction).astype(int)
    print(prediction)
    count = 0
    for i in range(train_size):
        print(prediction[i], y_train.iloc[i])
        if prediction[i] == y_train[i]:
            count += 1

    print("Accuracy : ", count / train_size * 100.0)


main()
