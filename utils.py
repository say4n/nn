#! /usr/bin/env python3

from imgcat import imgcat
import numpy as np

class Datapoint:
    def __init__(self, label, image):
        self.label = int(label)
        self.image = image

    def get_training_data(self):
        onehotlabel = np.zeros((10, 1))
        onehotlabel[self.label] = 1

        image = np.reshape(self.image, (self.image.size, 1))/255.0

        return image, onehotlabel

    def __repr__(self):
        imgcat(np.array(self.image).reshape(28,28))
        return f"Label:\t{self.label}"

class Dataset:
    def __init__(self, path, n_samples=float('inf')):
        self.n_samples = n_samples
        self.samples = self.load_dataset(path)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, position):
        return self.samples[position]

    def load_dataset(self, path):
        with open(path, "rt") as f:
            lines = map(lambda l: l.rstrip('\n').split(','), f.readlines())
            data = []

            counter = 0

            for line in lines:
                label, img = line[0], line[1:]
                img = np.array(list(map(int, img)))

                d = Datapoint(label, img)
                data.append(d)
                counter += 1

                if counter > self.n_samples:
                    break

            return data
