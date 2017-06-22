"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

            **  Headline Classifier  **

Auther(s): Gilad Lumbroso, Ady Kaiser, Omer Alon

===================================================
"""
import math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV  # This is the svm
import sklearn.svm as svm
import sklearn.neighbors as nb
from load_headlines import load_dataset
import polititians_classifier as pc


TRAIN_SET_SIZE = 0.6
VALIDATION_SET_SIZE = 0.3
TEST_SET_SIZE = 0.1


class Classifier(object):

    def classify(self,X):
        """
        Recieves a list of m unclassified headlines, and predicts for each one which newspaper published it.
        :param X: A list of length m containing the headlines' texts (strings)
        :return: y_hat - a binary vector of length m
        """


def shuffle(x, y):
    pairs_list = list(zip(x, y))
    random.shuffle(pairs_list)
    new_x, new_y = zip(*pairs_list)
    return new_x, new_y


def load_shuffled_data():
    x, y = load_dataset()
    return shuffle(x, y)


def repetitive_test(clf, x, y, num_of_tests):

    # Divide into train set and test set
    train_size = int(math.floor(TRAIN_SET_SIZE * len(y)))
    x_train = x[:train_size]
    y_train = y[:train_size]
    x_test = x[train_size:]
    y_test = y[train_size:]

    cul_score = 0
    for i in range(num_of_tests):
        shuffle(x_train, y_train)
        cul_score += clf.fit(x_train, y_train).score(x_test, y_test)

    return cul_score / num_of_tests


def import_politic2():
    politicians_USA_1 = list(pd.read_csv('USA.csv')['name'][:2865])
    politicians_UK = list(pd.read_csv('UK.csv')['name'][:2052])
    politicians_USA_2 = list(pd.read_csv('USA2.csv')['name'][:626])
    politicians_israel = list(pd.read_csv('KNESSET.csv')['name'][:2697])
    trump = list(pd.read_csv('trump.csv')['name'])
    x, y = load_shuffled_data()
    res = []
    for i in range(len(x)):
        counter = 0
        for j in range(len(politicians_israel)):
            if politicians_israel[j] in x[i]:
                counter += 1
        for k in range(len(politicians_USA_1)):
            if politicians_USA_1[k] in x[i]:
                counter += 1
        # for h in range(len(politicians_USA_2)):
        #     if politicians_USA_2[h] in x[i]:
        #         counter += 1
        # for v in range(len(politicians_UK)):
        #     if politicians_UK[v] in x[i]:
        #         counter += 1
        for u in range(len(trump)):
            if trump[u] in x[i]:
                counter += 5
        res.append([counter])
    clf_RFC = RandomForestClassifier(n_estimators=4)

    score_RFC = repetitive_test(clf_RFC, res, y, 400)
    print("RFC:", score_RFC)


def bag_of_words_classifier(x, y, test_x):

    # Transform the titles into vectors
    count_vect = CountVectorizer(ngram_range=(1, 2))
    X_train_counts = count_vect.fit_transform(x)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    X_test_counts = count_vect.transform(test_x)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    linear_svc = svm.LinearSVC().fit(X_train_tfidf, y)
    return linear_svc.predict(X_test_tfidf)


def length_classifier(x, y, test_x):
    x, y = load_shuffled_data()
    res = []
    test_res = []
    for headline in x:
        res.append([len(headline)])
    for headline in test_x:
        test_res.append([len(headline)])
    clf_NBR = nb.KNeighborsClassifier(40)
    clf_NBR.fit(res, y)
    return clf_NBR.predict(test_res)





def master_classifier():
    # Get Data
    x, y = load_shuffled_data()

    # Divide into train set and test set
    train_size = int(math.floor(TRAIN_SET_SIZE * len(y)))
    valid_size = int(math.floor((TRAIN_SET_SIZE + VALIDATION_SET_SIZE) * len(y)))
    x_train = x[:train_size]
    y_train = y[:train_size]
    x_valid = x[train_size:valid_size]
    y_valid = y[train_size:valid_size]
    x_test = x[valid_size:]
    y_test = y[valid_size:]

    # Get all the trained models
    # sc = sentiment_classifier(x_train, y_train)
    # lc_predictions = length_classifier(x_train, y_train, x_valid)
    # print("Length model ready.")
    # bofc_predictions = bag_of_words_classifier(x_train, y_train, x_valid)
    # print("Bag of words model ready.")
    poli_class = pc.PolititiansClassifier(x_train, y_train)
    print("Polititians model ready.")

    predictions = np.array([poli_class.predict(x_valid)])

    pred_vecs = predictions.T

    sgd = SGDClassifier()
    linear_svc = svm.LinearSVC()

    sgd.fit(pred_vecs, y_valid)

    x_test_ready = poli_class.prepare_samples(x_test)
    sgd_score = sgd.score(x_test_ready, y_test)

    # svc_score = linear_svc.fit(pred_vecs, y_valid).score(x_test, y_test)

    # sgd_score = repetitive_test(sgd, x_test, y, 100)
    # svc_score = repetitive_test(linear_svc, X_train_tfidf, y, 100)

    print("SGD: " + str(sgd_score))
    # print("Linear SVC: " + str(svc_score))


master_classifier()
