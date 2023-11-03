from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import random
from scipy.ndimage import zoom, rotate
import pickle
from tqdm import tqdm
from multiprocessing import Pool

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from keras import models, layers, losses, optimizers, Model
def pickle_dataset(zoom_factor=1):
    if not os.path.exists("pickles"):
        os.mkdir("pickles")

    for pi in range(9):
        pi_set = []
        mat_i = loadmat(f"mats/data-science-P{pi+1}.mat")
        coord_to_col = mat_i["meta"][0][0][8]
        data = mat_i["data"]
        info = mat_i["info"][0]

        for i in tqdm(range(359), desc=str(pi)):
            datum = data[i][0][0]

            scan = np.zeros((23, 61, 51))
            for x in range(51):
                for y in range(61):
                    for z in range(23):
                        scan[z, y, x] = datum[coord_to_col[x, y, z] - 1]

            mask = scan != mode(scan.flat).mode

            masked_scan = scan[mask]
            mask_mean, mask_std = np.mean(masked_scan), np.std(masked_scan)
            lt_mask, gt_mask = masked_scan < mask_mean, masked_scan > mask_mean
            std_left = np.sqrt(np.sum(np.square(masked_scan[lt_mask] - mask_mean)) / np.size(masked_scan[lt_mask]))
            std_right = np.sqrt(np.sum(np.square(masked_scan[gt_mask] - mask_mean)) / np.size(masked_scan[gt_mask]))

            scan[mask] -= mask_mean
            scan[mask][scan[mask] < mask_mean] /= std_left
            scan[mask][scan[mask] > mask_mean] /= std_right

            n = 3.5
            scan[scan > n * mask_std] = n * mask_std
            scan[scan < -n * mask_std] = -n * mask_std

            scan[scan == mode(scan.flat).mode] = scan[mask].min()
            scan = (scan - scan.min()) / (scan.max() - scan.min())
            scan = np.pad(scan, ((21, 20), (2, 1), (7, 6)))

            if zoom_factor > 1:
                scan = zoom(scan, zoom_factor, order=1)
            
            pi_set.append((scan, info[i][2][0]))

        with open(f"pickles/{pi}.pkl", "wb") as f:
            pickle.dump(pi_set, f)
        
        
            # w = 6
            # fig, ax = plt.subplots(w, w, constrained_layout=True)
            # fig.dpi = 100
            # bg_color = (225 / 255, 216 / 255, 226 / 255)
            # fig.set_facecolor(bg_color)
            
            # for j in range(w * w):
            #     ax[(j - j % w) // w, j % w].imshow(scan[j % 33], vmin=0, vmax=1, cmap="twilight")
            #     ax[(j - j % w) // w, j % w].set_xticks([])
            #     ax[(j - j % w) // w, j % w].set_yticks([])
            #     plt.setp(ax[(j - j % w) // w, j % w].spines.values(), color=bg_color)
            # plt.show()

# pickle_dataset()
pickles = [pickle.load(open(f"pickles/{i}.pkl", "rb")) for i in range(1)]
pickles = [pickle.load(open(f"pickles/{i}.pkl", "rb")) for i in range(1)]
targets = set([pickles[0][i][1] for i in range(len(pickles[0]))])
targets = {k: v for k, v in zip(targets, [[1 if i == j else 0 for i in range(60)] for j in range(len(targets))])}
def get_sample(i):
    p = random.choice(pickles)
    scan, target = random.choice(p)

    angles = np.random.randint(low=-5, high=5, size=(3, ))
    scan = rotate(scan, angles[0], (0, 1), reshape=False)
    scan = rotate(scan, angles[1], (1, 2), reshape=False)
    scan = rotate(scan, angles[2], (2, 0), reshape=False)
        
    scan = np.expand_dims(scan, -1)
    target = targets[target]

    return scan, target
def get_batch(batch_size=100):
    samples = list(tqdm(Pool(processes=4).imap(get_sample, range(batch_size)), total=batch_size))

    batch_x = [sample[0] for sample in samples]
    batch_y = [sample[1] for sample in samples]
    
    return batch_x, batch_y

def get_model():
    inputs = layers.Input((*pickles[0][0][0].shape, 1))
    x = layers.Conv3D(32, 16, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv3D(32, 8, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv3D(32, 8, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv3D(32, 4, strides=2, padding="same", activation="relu")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(60, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    
    return model
model = get_model()
model.compile(optimizers.Adam(), losses.CategoricalCrossentropy())

def train(num_epochs=100, num_samples=1000):
    batches_x, batches_y = get_batch(num_samples)

    batches_x, batches_y = np.array(batches_x), np.array(batches_y)
    model.fit(batches_x, batches_y, epochs=num_epochs, batch_size=16)

train()