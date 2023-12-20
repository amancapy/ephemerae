import seaborn as sns
import matplotlib.pyplot as plt
import json


distmat = json.loads(open("leave2/dist_mat.json", "r").read())
confmat = json.loads(open("leave2/conf_mat.json", "r").read())


ax = sns.heatmap(distmat, cmap=sns.diverging_palette(20, 220, n=200), square=True)
plt.show()