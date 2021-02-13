import nltk
import pandas as pd
import numpy as np
from collections import Counter
from numpy import random

def tokenize(text):
    '''
        Generic wrapper around different tokenization methods.
    '''
    return nltk.TweetTokenizer(strip_handles=True, preserve_case=False).tokenize(text)

def get_representation(toate_cuvintele, how_many):
    '''
        Extract the first most common words from a vocabulary
        and return two dictionaries: word to index and index to word
               @  che  .   ,   di  e
        text0  0   1   0   2   0   1
        text1  1   2 ...
        ...
        textN  0   0   1   1   0   2
    '''
    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd

def get_corpus_vocabulary(corpus):
    '''
        Write a function to return all the words in a corpus.
    '''
    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter

def text_to_bow(text, wd2idx):
    '''
        Convert a text to a bag of words representation.
               @  che  .   ,   di  e
        text   0   1   0   2   0   1
    '''
    features = np.zeros(len(wd2idx))
    for token in tokenize(text):
        if token in wd2idx:
            features[wd2idx[token]] += 1
    #normalizare l2
    norm = 0
    for i in range(0, len(features)):
        norm += features[i] ** 2
    norm = np.sqrt(norm)
    for i in range(0, len(features)):
        if norm != 0:
            features[i] = features[i] / norm
        else:
            features[i] = features[i]
    return features

def corpus_to_bow(corpus, wd2idx):
    '''
        Convert a corpus to a bag of words representation.
               @  che  .   ,   di  e
        text0  0   1   0   2   0   1
        text1  1   2 ...
        ...
        textN  0   0   1   1   0   2
    '''
    all_features = []
    for text in corpus:
        all_features.append(text_to_bow(text, wd2idx))
    all_features = np.array(all_features)
    return all_features

def write_prediction(out_file, predictions):
    '''
        A function to write the predictions to a file.
        id,label
        5001,1
        5002,1
        5003,1
        ...
    '''
    with open(out_file, 'w') as fout:
        # aici e open in variabila 'fout'
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)
    # aici e fisierul closed

def kfold(k, data, labels):
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    fold_size = int(len(labels) / k)
    for i in range(0, len(labels), fold_size):
        valid = data[indici[i:i + fold_size]]
        y_valid = labels[indici[i:i + fold_size]]
        parte1 = indici[:i]
        parte2 = indici[i + fold_size:]
        indici_train = np.concatenate([parte1, parte2])
        train = data[indici_train]
        y_train = labels[indici_train]
        yield train, valid, y_train, y_valid

def fscore(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    acc = 0
    precision = 0
    recall = 0
    for idx in range(0, len(y_true)):
        if y_true[idx] == 1 and y_pred[idx] == 1:
            tp += 1
        if y_true[idx] == 0 and y_pred[idx] == 1:
            fp += 1
        if y_true[idx] == 1 and y_pred[idx] == 0:
            fn += 1
        if y_true[idx] == 0 and y_pred[idx] == 0:
            tn += 1
    if (tp + tn + fn + fp) > 0:
        acc = (tp + tn) / (tp + tn + fp + fn)
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    fscore = 2 * precision * recall/(precision + recall)
    return tp, tn, fp, fn, acc, fscore



train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']
toate_cuvintele = get_corpus_vocabulary(corpus)
wd2idx, idx2wd = get_representation(toate_cuvintele, 70)
data = corpus_to_bow(corpus, wd2idx)
labels = train_df['label']
test_data = corpus_to_bow(test_df['text'], wd2idx)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 7)
# scrierea predictiilor (scor 75.225% kaggle)
clf.fit(data, labels)
preds = clf.predict(test_data)
write_prediction('352_TroianStefan_submisie1.csv', preds)



# # kfold si matrice de confuzie
# predictie_medie = []
# tp = []
# tn = []
# fp = []
# fn = []
# acc = []
#
# i = 1
# for train, valid, y_train, y_valid in kfold(10, train_df['text'], train_df['label']):
#     corpus = train
#     toate_cuvintele = get_corpus_vocabulary(corpus)
#     wd2idx, idx2wd = get_representation(toate_cuvintele, 70)
#     data = corpus_to_bow(corpus, wd2idx)
#     test_data = corpus_to_bow(valid, wd2idx)
#     clf.fit(data, y_train)
#     preds = clf.predict(test_data)
#
#     # initializare variabile pentru matricea de confuzie si fscore
#     tp_local, tn_local, fp_local, fn_local, acc_local, fscore_local = fscore(np.array(y_valid), preds)
#     print(str(i) + "\tAcc: " + str(acc_local) + "\tFs: " + str(fscore_local))
#     i += 1
#     tp.append(tp_local)
#     tn.append(tn_local)
#     fp.append(fp_local)
#     fn.append(fn_local)
#     acc.append(acc_local)
#     predictie_medie.append(fscore_local)
#
# #afisare acuratete si fscore
# print("Acc: " + str(np.mean(acc)) + "\t Fs:" + str(np.mean(predictie_medie)))
#
# #afisare matrice de confuzie
# matrix_confusion = np.zeros((2,2))
# matrix_confusion[0][0] = np.sum(tp)
# matrix_confusion[0][1] = np.sum(fn)
# matrix_confusion[1][0] = np.sum(fp)
# matrix_confusion[1][1] = np.sum(tn)
# print(matrix_confusion)


