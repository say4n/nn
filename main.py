#! /usr/bin/env python3

import logging
import nn
import numpy as np
import utils

logging.basicConfig(level=logging.ERROR)

dataset = {
    "train": utils.Dataset("data/mnist_train.csv", 1000),
    "test": utils.Dataset("data/mnist_test.csv", 100)
}


INPUT_DIM = 28*28
OUTPUT_DIM = 10

LAYERS = [(INPUT_DIM, 256), (256, 64), (64, 10)]

net = nn.Network("MNIST_Net")
for layer in LAYERS:
    net.add_layer(*layer)

print(net)

for epoch in range(10):
    avg_loss = 0
    for idx, data in enumerate(dataset["train"]):
        x, y = data.get_training_data()

        y_hat = net.forward_propagate(x)            # predict
        net.backward_propagate(y_hat, y)            # backprop
        net.update_parameters()                     # update

        loss = np.sum((y_hat - y)**2)
        avg_loss = (avg_loss * (idx) + loss)/(idx + 1)
        progress = 100 * (idx+1) / len(dataset['train'])

        print(f"\r[Epoch #{epoch+1}] [Progress: {progress:.2f}%] [Loss: {loss:.8f}]", end="")
    print(f"\r[Epoch #{epoch+1}] [Progress: 100.00%] [Average Loss: {avg_loss:.8f}]")

accurate = 0
for data in dataset["test"]:
    x, _ = data.get_training_data()
    predict = net.predict(x)

    if np.argmax(predict) == data.label:
        accurate += 1

    print(f"[True Label, Predicted Label: {data.label}, {np.argmax(predict)}]")

accuracy = 100 * accurate/len(dataset['test'])
print(f"[Accuracy: {accuracy:.2f}%]")
