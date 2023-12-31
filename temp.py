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


bat = tf.random.normal((64, 50), 0, 1)
# bat = bat / tf.norm(bat, axis=-1, keepdims=True)

for v in bat:
    print(v, tf.norm(v))