#! /usr/bin/env python3

from copy import deepcopy
import logging
import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return np.where(x > 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1+np.exp(x)))

def d_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

class Network:
    def __init__(self, name="Network", lr=0.003):
        self.name = name
        self.lr = lr
        self.n_layers = 0
        self.weights = []
        self.biases = []
        self.activations = [None]
        self.errors = []
        self.gradients = []

    def add_layer(self, input_dim, output_dim):
        self.weights.append(np.array(np.random.rand(input_dim,
                                                   output_dim),
                                    dtype=np.float128))
        self.biases.append(np.array(np.random.rand(output_dim, 1),
                                    dtype=np.float128))
        self.activations.append(None)
        self.errors.append(None)
        self.gradients.append(None)

        self.n_layers += 1

    def predict(self, data):
        intermediate = data
        for idx, layer in enumerate(self.weights):
            intermediate = sigmoid(np.dot(layer.T, intermediate) + self.biases[idx])

        return softmax(intermediate)

    def forward_propagate(self, data):
        intermediate = deepcopy(data)
        self.activations[0] = intermediate

        for idx, layer in enumerate(self.weights):
            intermediate = np.dot(layer.T, intermediate) + self.biases[idx]
            self.activations[idx+1] = intermediate
            intermediate = sigmoid(intermediate)

        return softmax(intermediate)

    def backward_propagate(self, y_hat, y):
        error = (y - y_hat)
        pred = y_hat

        for idx in range(len(self.weights)-1, 0, -1):
            self.errors[idx] = error
            pred = np.dot(self.weights[idx], error)
            error = (pred - self.activations[idx])

        self.errors[0] = error

    def update_parameters(self):
        for idx in range(self.n_layers):
            # weights
            ds_w = d_sigmoid(self.activations[idx])
            delta_w = -self.lr * np.dot(ds_w, self.errors[idx].T) * self.activations[idx]

            # biases
            ds_b = d_sigmoid(self.biases[idx])
            delta_b = -self.lr * np.dot(ds_b.T, self.errors[idx])

            # update wrt gradients
            self.weights[idx] -= delta_w
            self.biases[idx] -= delta_b

    def __repr__(self):
        txt = f"({self.name}) Layers:\n"
        for idx, layer in enumerate(self.weights):
            txt += f" Layer_{idx}{layer.shape}\n"

        return txt

