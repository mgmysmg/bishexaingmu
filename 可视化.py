# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:28:28 2023

@author: 28450
"""
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


train = pd.read_csv('train.txt')
texts = list(train['text'])
ndims = 100
model = Word2Vec(sentences=texts, vector_size=ndims, window=5)
total = len(texts)
vecs = np.zeros([total, ndims])
for i, sentence in enumerate(texts):
    counts, row = 0, 0
    for char in sentence:
        try:
            if char != ' ':
                row += model.wv[char]
                counts += 1
        except:
            pass
    if counts == 0:
        print(sentence)
    vecs[i, :] = row / counts
x_train = vecs
x_tsne = TSNE(n_components=2, learning_rate=500,random_state=0).fit_transform(x_train)
plt.figure(figsize=(8, 8))
colors = list(map(lambda x: 'red' if x == 1 else 'blue', train['y']))
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=colors, alpha=0.2, s=30, lw=0)
print('Word2Vec: 白话文(蓝色)与文言文(红色)')
plt.show()