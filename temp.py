import numpy as np
import matplotlib.pyplot as plt

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

coeff_vecs = get_coeff_vecs()
verbs = [k for k in coeff_vecs["house"]]
coeff_vecs = {k: [coeff_vecs[k][verb] for verb in verbs] for k in coeff_vecs}
coeff_vecs = np.array([coeff_vecs[k] for k in coeff_vecs])

plt.plot(np.sum(coeff_vecs, axis=0))
plt.show()