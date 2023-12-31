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

for k in coeff_vecs:
    print(coeff_vecs[k], np.linalg.norm(coeff_vecs[k]))