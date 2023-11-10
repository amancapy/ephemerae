import pickle

def get_words():
    with open("coeff_vecs.txt", "r") as f:
        f = f.read().split("\n\n\n")
        noun_dict = {}
        for noun_vec in f:
            split = noun_vec.split("\n\n")
            noun = split[0].split()[2][:-1]
            
            vec = split[1]
            vec = [item.strip().replace("(", "").replace(")", "") for item in vec.split(",\n")]
            vec = [(item.split()[:-1], item.split()[-1]) for item in vec]

            vec = sorted(vec, key=lambda x: x[0])
            vec = {" ".join(item[0]): float(item[1]) for item in vec}
            noun_dict[noun] = vec

            verbs = list(vec.keys())
            verbs = [item.split() for item in verbs]
            verbs = [item for sublist in verbs for item in sublist]

    return list(noun_dict.keys()), verbs


nouns, verbs = get_words()

wordset = set(nouns + verbs)
vecs = {}
with open("glove.6B/glove.6B.300d.txt", "r") as f:
    f = f.readlines()
    for l in f:
        l = l.split()
        if l[0] in wordset:
            vecs[l[0]] = [float(item) for item in l[1:]]

print(len(vecs))

with open("vecs.pkl", "wb") as f:
    pickle.dump(vecs, f)