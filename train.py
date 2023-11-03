import pickle
import tensorflow as tf
from keras import models, losses, optimizers, layers, Model
from scipy.ndimage import zoom, rotate
import random
import numpy as np
from tqdm import tqdm

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


pickles = [pickle.load(open(f"pickles/{i}.pkl", "rb")) for i in range(1)]
targets = set([pickles[0][i][1] for i in range(len(pickles[0]))])
targets = {k: v for k, v in zip(targets, [[1 if i == j else 0 for i in range(60)] for j in range(len(targets))])}


def get_batch(batch_size=100):
    batch_x, batch_y = [], []
    for i in tqdm(range(batch_size), leave=False, ncols=100):
        p = random.choice(pickles)
        scan, target = random.choice(p)

        angles = np.random.randint(low=-5, high=5, size=(3, ))
        scan = rotate(scan, angles[0], (0, 1), reshape=False)
        scan = rotate(scan, angles[1], (1, 2), reshape=False)
        scan = rotate(scan, angles[2], (2, 0), reshape=False)
        
        scan = np.expand_dims(scan, -1)
        
        batch_x.append(scan)
        batch_y.append(targets[target])

    return batch_x, batch_y


def get_model():
    inputs = layers.Input((*pickles[0][0][0].shape, 1))
    x = layers.Conv3D(32, 16, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv3D(32, 8, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv3D(32, 8, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv3D(32, 4, strides=2, padding="same", activation="relu")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(60, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    
    return model


model = get_model()
model.compile(optimizers.Adam(), losses.MeanSquaredError())

def train(num_epochs=100, num_samples=1000):
    batches_x, batches_y = get_batch(num_samples)

    batches_x, batches_y = np.array(batches_x), np.array(batches_y)
    model.fit(batches_x, batches_y, epochs=num_epochs, batch_size=16)

train()