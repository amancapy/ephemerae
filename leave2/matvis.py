import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np


distmat = np.array(json.loads(open("leave2/dist_mat.json", "r").read()))
confmat = np.array(json.loads(open("leave2/conf_mat.json", "r").read()))
print(1 - np.count_nonzero(confmat) / 3600)

ax = sns.heatmap(distmat, cmap=sns.diverging_palette(20, 220, n=200), square=True)
plt.show()