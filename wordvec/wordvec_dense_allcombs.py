import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle
from tqdm import tqdm
import gc
from itertools import combinations
import json

from multiprocessing import Pool, Manager
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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


def l2(a, b):
    return norm(np.subtract(a, b))


batch_size = 64
all_combinations = list(combinations(range(60), 2))[:4]
dist_mat = {i: {j: 0 for j in range(60)} for i in range(60)}
dist_mat = Manager().dict(dist_mat)


def train_test_over_these_indices(inp):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.get_logger().setLevel("ERROR")


    class BasisSum(Model):
        def __init__(self):
            super().__init__()
            self.basis = tf.Variable(tf.convert_to_tensor([verbs[verb] for verb in verbs]), trainable=False, name="verb_basis")
            self.d1 = layers.Dense(64, activation="relu")
            self.d2 = layers.Dense(32, activation="relu")
            self.dn = layers.Dense(self.basis.shape[0], activation="sigmoid")
            
        @tf.function(reduce_retracing=True)
        def call(self, x):
            x = self.d1(x)
            x = self.d2(x)
            x = self.dn(x)
            x = x / tf.reduce_sum(x, axis=-1, keepdims=True)
            
            x = tf.einsum("bi,ij->bj", x, self.basis)
            
            return x
    
    idxs, j = inp
    correct_count = 0

    x = np.array([item[0] for item in pickles])
    y = [item[1] for item in pickles]
    y = np.array([nouns[item] for item in y])

    x, y = tf.cast(x, tf.dtypes.float32), tf.cast(y, tf.dtypes.float32)
    
    pbar = tqdm(idxs, position=j)
    for i, comb in enumerate(pbar):
        comp = sorted(list(set.difference(set(range(60)), set(comb))))

        model = BasisSum()
        loss = losses.MeanSquaredError()
        opt = optimizers.Adam(0.001)
        
        train_x, test_x = tf.gather(x, comp), tf.gather(x, comb)
        train_y, test_y = tf.gather(y, comp), tf.gather(y, comb)

        for j in range(5000):
            idx1 = tf.random.uniform(shape=[batch_size], minval=0, maxval=tf.shape(train_x)[0], dtype=tf.int32)
            idx2 = tf.random.uniform(shape=[batch_size], minval=0, maxval=tf.shape(train_x)[0], dtype=tf.int32)

            batch_x1, batch_y1 = tf.gather(train_x, idx1), tf.gather(train_y, idx1)
            batch_x2, batch_y2 = tf.gather(train_x, idx2), tf.gather(train_y, idx2)

            ratios = tf.random.uniform((len(batch_x1), 1), 0, 1)
            batch_x = batch_x1 * ratios + batch_x2 * (1 - ratios)
            batch_y = batch_y1 * ratios + batch_y2 * (1 - ratios)

            with tf.GradientTape() as tape:
                pred_y = model(batch_x)
                batchloss = loss(batch_y, pred_y)
                grads = tape.gradient(batchloss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))

        pred = model(test_x)
        t1, t2 = test_y.numpy()
        t1, t2 = t1.flat, t2.flat
        p1, p2 = pred.numpy()
        p1, p2 = p1.flat, p2.flat

        t1_to_p1 = l2(t1, p1)
        t1_to_p2 = l2(t1, p2)
        t2_to_p1 = l2(t2, p1)
        t2_to_p2 = l2(t2, p2)

        dist_mat[comb[0]][comb[0]] += t1_to_p1
        dist_mat[comb[0]][comb[1]] += t1_to_p2
        dist_mat[comb[1]][comb[0]] += t2_to_p1
        dist_mat[comb[1]][comb[1]] += t2_to_p2

        correct = t1_to_p1 + t2_to_p2
        incorrect = t1_to_p2 + t2_to_p1

        correct_count += int(correct < incorrect)

        pbar.set_description(f"accuracy: {correct_count / (i + 1):.3f}")


pool = Pool()
chunks = np.array_split(all_combinations, 4)
pool.map(train_test_over_these_indices, [(chunk, i) for i, chunk in enumerate(chunks)])
open("dist_mat.json", "w+").write(json.dumps(dict(dist_mat)))