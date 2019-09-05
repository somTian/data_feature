from gensim.models.keyedvectors import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('data/vectors1.bin',binary=True)

# print(word_vectors['动力'])
# print(word_vectors.wv.most_similar(positive=['座椅','柔软']))

import numpy as np
def Odistanct(vec1,vec2):
    dist = np.linalg.norm(vec1-vec2)
    return dist

vec1 = word_vectors['座椅']
vec2 = word_vectors['舒适']

dist = Odistanct(vec1,vec2)
print(dist)