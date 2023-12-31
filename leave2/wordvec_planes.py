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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras import models, layers, losses, optimizers, regularizers, Model


brain_index = pickle.load(open("data/support.pkl", "rb"))
nouns, verbs = pickle.load(open("data/vecs.pkl", "rb"))

pickles = [pickle.load(open(f"data/pickles/{i}.pkl", "rb")) for i in range(6)]
pickles = [item for sublist in pickles for item in sublist]
pickles = sorted(pickles, key=lambda x: x[1])
pickles = [[item for item in pickles if item[1] == noun] for noun in nouns]
pickles = [([item[0] for item in sublist], sublist[0][1]) for sublist in pickles]
pickles = [([item[brain_index] for item in sublist[0]], sublist[1]) for sublist in pickles]


def l2(a, b):
    return norm(np.subtract(a, b))


batch_size = 64

all_combinations = list(combinations(range(60), 2))
dist_mat = Manager().dict({(i, j): 0 for i in range(60) for j in range(60)})
conf_mat = Manager().dict({(i, j): 0 for i in range(60) for j in range(60)})


count = Value("i", 0)
correct_count = Value("i", 0)

x = np.stack([item[0] for item in pickles], axis=0)
y = [item[1] for item in pickles]
y = [nouns[item] for item in y]
y = np.array([[item for _ in range(x.shape[1])] for item in y])


class BasisSum(Model):
    def __init__(self):
        super().__init__()
        self.basis = tf.Variable(tf.convert_to_tensor([verbs[verb] for verb in verbs]), trainable=False, name="verb_basis")

        self.d1 = layers.Dense(64, activation="relu")
        self.d2 = layers.Dense(32, activation="relu")
        self.dn = layers.Dense(25, activation="sigmoid")
        
    @tf.function(reduce_retracing=True)
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.dn(x)
        x /= tf.reduce_sum(x, axis=-1, keepdims=True)

        x = tf.einsum("bi,ij->bj", x, self.basis)
        
        return x
            
def train_test_over_these_indices(idxs):
    gc.collect()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.get_logger().setLevel("ERROR")

            
    comp = [i for i in range(60) if i not in idxs]
    
    train_x, test_x = tf.gather(x, comp), tf.gather(x, idxs)
    train_x, test_x = tf.reshape(train_x, (train_x.shape[0] * train_x.shape[1], train_x.shape[2])), tf.reduce_sum(test_x, axis=1) / test_x.shape[0]
    train_y, test_y = tf.gather(y, comp), tf.gather(y, idxs)
    train_y, test_y = tf.reshape(train_y, (train_y.shape[0] * train_y.shape[1], train_y.shape[2])), tf.reduce_sum(test_y, axis=1) / test_y.shape[0]

    train_x, train_y, test_x, test_y = tf.cast(train_x, tf.dtypes.float32), tf.cast(train_y, tf.dtypes.float32), tf.cast(test_x, tf.dtypes.float32), tf.cast(test_y, tf.dtypes.float32)
    idx1 = tf.random.uniform(shape=[batch_size], minval=0, maxval=tf.shape(train_x)[0], dtype=tf.int32)
    idx2 = tf.random.uniform(shape=[batch_size], minval=0, maxval=tf.shape(train_x)[0], dtype=tf.int32)

    batch_x1, batch_y1 = tf.gather(train_x, idx1), tf.gather(train_y, idx1)
    batch_x2, batch_y2 = tf.gather(train_x, idx2), tf.gather(train_y, idx2)

    ratios = tf.random.uniform((len(batch_x1), 1), 0, 1)
    batch_x = batch_x1 * ratios + batch_x2 * (1 - ratios)
    batch_y = batch_y1 * ratios + batch_y2 * (1 - ratios)
    

    model = BasisSum()
    loss = losses.MeanSquaredError()
    opt = optimizers.Adam(0.001)

    batchlosses = []
    for j in range(5000):

        with tf.GradientTape() as tape:
            pred_y = model(batch_x)
            batchloss = loss(batch_y, pred_y)
            grads = tape.gradient(batchloss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            batchlosses.append(float(batchloss))
        
        if j % 100 == 0:
            ...
            # print(j, sum(batchlosses[-100:]) / 100)

    pred = model(test_x)
    t1, t2 = test_y.numpy()
    t1, t2 = t1.flat, t2.flat
    p1, p2 = pred.numpy()
    p1, p2 = p1.flat, p2.flat

    t1_to_p1 = l2(t1, p1)
    t1_to_p2 = l2(t1, p2)
    t2_to_p1 = l2(t2, p1)
    t2_to_p2 = l2(t2, p2)

    if t1_to_p2 < t1_to_p1:
        conf_mat[(idxs[0], idxs[1])] = t1_to_p1 / t1_to_p2
    
    if t2_to_p1 < t2_to_p2:
        conf_mat[(idxs[1], idxs[0])] = t2_to_p2 / t2_to_p1

    dist_mat[(idxs[0], idxs[0])] += t1_to_p1
    dist_mat[(idxs[0], idxs[1])] += t1_to_p2
    dist_mat[(idxs[1], idxs[0])] += t2_to_p1
    dist_mat[(idxs[1], idxs[1])] += t2_to_p2

    correct = t1_to_p1 + t2_to_p2
    incorrect = t1_to_p2 + t2_to_p1

    correct_count.value += int(correct < incorrect)
    count.value += 1


processes = 3
chunks = np.array_split(range(1770), 1770 // processes)

for chunk in chunks:
    print(chunk[0], end=" ")

    t = ptime()
    Pool(processes=processes).map(train_test_over_these_indices, all_combinations[chunk[0]:chunk[-1]+1])
    print(f"{correct_count.value}/{count.value} == {correct_count.value / (count.value)}", ptime() - t)


dist_mat_ = [[0 for _ in range(60)] for _ in range(60)]
for idxs in dist_mat:
    dist_mat_[idxs[0]][idxs[1]] += dist_mat[idxs]

for i in range(60):
    dist_mat_[i][i] = dist_mat_[i][i] / 60
    
conf_mat_ = [[0 for _ in range(60)] for _ in range(60)]
for idxs in conf_mat:
    conf_mat_[idxs[0]][idxs[1]] += conf_mat[idxs]

dist_mat = dist_mat_
conf_mat = conf_mat_
del dist_mat_, conf_mat_

open("leave2/dist_mat_planes.json", "w+").writelines(json.dumps(dist_mat))
open("leave2/conf_mat_planes.json", "w+").writelines(json.dumps(conf_mat))