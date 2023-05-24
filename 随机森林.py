# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 14:54:48 2023

@author: 28450
"""
import pandas as pd
import numpy as np
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec


train_all = pd.read_csv('train.txt')
sanguo = pd.read_csv('sanguo.txt')
simaqian = pd.read_csv('simaqian.txt')
ming = pd.read_csv('ming.txt')
songshi = pd.read_csv('songshi.txt')
train, test_all = train_test_split(train_all, test_size=0.2, random_state=0)
train = copy.deepcopy(train)
test_all = copy.deepcopy(test_all)
val, test = train_test_split(test_all, test_size=0.5, random_state=0)
val = copy.deepcopy(val)
test = copy.deepcopy(test)
submit = copy.deepcopy(test[['id', 'y']])
n_total = len(train) + len(val) + len(test)
n_train = len(train)
n_val = len(val)
n_test = len(test)
labeled_texts = []
texts = list(train['text']) + list(val['text']) + list(test['text']) + list(ming['text']) + list(sanguo['text']) + list(simaqian['text'])  + list(songshi['text'])
ndims = 100
model = Word2Vec(sentences=texts, vector_size=ndims, window=5, workers=5,sg=1)
vecs = np.zeros([n_total, ndims])
for i, sentence in enumerate(texts[:n_total]):
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
best_max_depth = 2
best_logloss = 1000
print("左为max_depth值 右为log-loss值：")
for d in range(2,21):
    clf = RandomForestClassifier(max_depth=d, random_state=0)
    clf.fit(vecs[:n_train], train['y'])
    val['pred'] = (clf.predict_proba(vecs[n_train: (n_train + n_val)])[:, 1] > 0.5).astype(int)
    val['prob'] = clf.predict_proba(vecs[n_train: (n_train + n_val)])[:, 1]
    y_true = val['y']
    y_pred = val['prob']
    logloss = log_loss(y_true, y_pred)
    print(d,"    ",logloss)
    if logloss < best_logloss : 
        best_logloss = logloss
        best_max_depth = d
print("max_depth最佳值为：",best_max_depth,end="")
print("，此时log-loss值为:",best_logloss)
clf = RandomForestClassifier(max_depth=best_max_depth, random_state=0)
clf.fit(vecs[:n_train], train['y'])
submit['y'] = clf.predict_proba(vecs[(n_train + n_val):(n_train + n_val + n_test)])[:, 1]
submit.to_csv('my_prediction.csv', index=False)
test['pred'] = (submit['y'] > 0.5).astype(int)
test['prob'] = clf.predict_proba(vecs[(n_train + n_val):(n_train + n_val + n_test)])[:, 1]
y_true = test['y']
y_pred = test['prob']
print("测试集log-loss值为：",log_loss(y_true, y_pred))
