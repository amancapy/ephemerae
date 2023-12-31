import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


norms = [np.linalg.norm([float(item) for item in l.split()[1:]]) for l in open("data/glove.6B/glove.6B.50d.txt", "r").readlines()]

plt.hist(norms, bins=1000)
plt.show()