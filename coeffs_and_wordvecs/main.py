import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle
from tqdm import tqdm
import gc
from itertools import combinations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras import models, layers, losses, optimizers, regularizers, Model


def get_coeff_vecs():
    with open("data/coeff_vecs.txt", "r") as f:
        f = f.read().split("\n\n\n")
        noun_dict = {}
        for noun_vec in f:
            split = noun_vec.split("\n\n")
            noun = split[0].split()[2][:-1]
            
            vec = split[1]
            vec = [item.strip().replace("(", "").replace(")", "") for item in vec.split(",\n")]
            vec = [(item.split()[:-1], item.split()[-1]) for item in vec]

            vec = sorted(vec, key=lambda x: x[0])
            vec = {item[0][0]: float(item[1]) for item in vec}

            noun_dict[noun] = vec

    return noun_dict


brain_index = pickle.load(open("data/support.pkl", "rb"))
noun_vecs, verb_vecs = pickle.load(open("data/vecs.pkl", "rb"))

coeff_vecs = get_coeff_vecs()
coeff_vecs = {k: [coeff_vecs[k][noun] for noun in verb_vecs] for k in coeff_vecs}

pickles = [pickle.load(open(f"data/pickles/{i}.pkl", "rb")) for i in range(1)]
pickles = [item for sublist in pickles for item in sublist]
pickles = sorted(pickles, key=lambda x: x[1])
pickles = [[item for item in pickles if item[1] == noun] for noun in noun_vecs]
pickles = [(np.add.reduce([item[0] for item in sublist]) / len(sublist), sublist[0][1]) for sublist in pickles]
pickles = [(item[0][brain_index], item[1]) for item in pickles]


def l2(a, b):
    return norm(np.subtract(a, b))


class CoeffPred(Model):
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l1_l2(0.03, 0.03), bias_regularizer=regularizers.l1_l2(0.03, 0.03))
        self.d2 = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l1_l2(0.03, 0.03), bias_regularizer=regularizers.l1_l2(0.03, 0.03))
        self.dn = layers.Dense(25, activation="sigmoid", kernel_regularizer=regularizers.l1_l2(0.03, 0.03), bias_regularizer=regularizers.l1_l2(0.03, 0.03))
        
    @tf.function(reduce_retracing=True)
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.dn(x)
        x /= tf.norm(x, axis=0, keepdims=True)
        return x
    

batch_size = 64
pbar = tqdm(combinations(range(60), 58), total=1770)
correct_count = 0

x = np.array([item[0] for item in pickles])
y = [item[1] for item in pickles]
y = np.array([coeff_vecs[item] for item in y])

x, y = tf.cast(x, tf.dtypes.float32), tf.cast(y, tf.dtypes.float32)
basis = tf.convert_to_tensor([verb_vecs[verb] for verb in verb_vecs])


for i, comb in enumerate(pbar):
    gc.collect()
    comp = list(set.difference(set(range(60)), set(comb)))

    model = CoeffPred()
    loss = losses.MeanSquaredError()
    opt = optimizers.Adam(0.0005)
    
    train_x, test_x = tf.gather(x, comb), tf.gather(x, comp)
    train_y, test_y = tf.gather(y, comb), tf.gather(y, comp)

    batchlosses = []
    for j in range(1500):
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

            batchlosses.append(float(batchloss))

        if j % 100 == 0:
            # print(j, sum(batchlosses[-100:]) / 100)
            ...

    pred_y = model(test_x)
    t1, t2 = tf.einsum("bi,ij->bj", test_y, basis).numpy()
    t1, t2 = t1.flat, t2.flat
    p1, p2 = tf.einsum("bi,ij->bj", pred_y, basis).numpy()
    p1, p2 = p1.flat, p2.flat
    
    correct = l2(t1, p1) + l2(t2, p2)
    incorrect = l2(t1, p2) + l2(t2, p1)

    correct_count += int(correct < incorrect)

    pbar.set_description(f"accuracy: {correct_count / (i + 1):.3f}")