import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle
from time import perf_counter as ptime
from itertools import combinations
import json
from multiprocessing import Pool, Manager, Value
import os
import gc
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras import models, layers, losses, optimizers, regularizers, Model



brain_index = pickle.load(open("data/support.pkl", "rb"))
nouns, verbs = pickle.load(open("data/vecs.pkl", "rb"))

pickles = [pickle.load(open(f"data/pickles/{i}.pkl", "rb")) for i in range(1)]
pickles = [item for sublist in pickles for item in sublist]
pickles = sorted(pickles, key=lambda x: x[1])
pickles = [[item for item in pickles if item[1] == noun] for noun in nouns]
pickles = [(np.add.reduce([item[0] for item in sublist]) / len(sublist), sublist[0][1]) for sublist in pickles]
pickles = [(item[0][brain_index], item[1]) for item in pickles]


x = np.array([item[0] for item in pickles])
y = [item[1] for item in pickles]
y = np.array([nouns[item] for item in y])

x, y = tf.cast(x, tf.dtypes.float32), tf.cast(y, tf.dtypes.float32)

batch_size = 64

def get_batch(x, y):
    batch_x = tf.tile(tf.expand_dims(x, 0), (batch_size, 1, 1))
    batch_y = tf.tile(tf.expand_dims(y, 0), (batch_size, 1, 1))

    ratios = tf.random.normal((batch_size, x.shape[0]), 0, 10)
    ratios = tf.math.softmax(ratios)

    batch_x = tf.einsum("ijk, ij -> ik", batch_x, ratios)
    batch_y = tf.einsum("ijk, ij -> ik", batch_y, ratios)

get_batch(x, y)