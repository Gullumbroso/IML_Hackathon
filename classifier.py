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
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV  # This is the svm
import sklearn.svm as svm
from load_headlines import load_dataset
import load_headlines
from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestClassifier


TRAIN_SET_SIZE = 0.8


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


def length_array():
    x, y = load_shuffled_data()
    train_size = int(0.8 * len(x))
    res = []
    for headline in x:
        res.append([len(headline)])
    train_set = res[:train_size]
    train_label = y[:train_size]
    valid_set = res[train_size:]
    valid_label = y[train_size:]
    clf_SVC = svm.SVC()
    clf_NBR = neighbors.KNeighborsClassifier(40)
    clf_RFC = RandomForestClassifier(n_estimators=4)
    score_SVC = clf_SVC.fit(train_set, train_label).score(valid_set, valid_label)
    score_NBR = clf_NBR.fit(train_set, train_label).score(valid_set, valid_label)
    score_RFC = clf_RFC.fit(train_set, train_label).score(valid_set, valid_label)
    print("SVC:", score_SVC)
    print("NBR:", score_NBR)
    print("RFC:", score_RFC)


def train_google(language_client):
    x, y = load_shuffled_data()
    train_size = int(0.8 * len(x))
    res = []
    for headline in x:
        document = language_client.document_from_text(headline)
        sentiment = document.analyze_sentiment().sentiment
        res.append([sentiment.score, sentiment.magnitude])

    train_set = res[:train_size]
    train_label = y[:train_size]
    valid_set = res[train_size:]
    valid_label = y[train_size:]

    clf_SVC = svm.SVC()
    clf_NBR = neighbors.KNeighborsClassifier(40)
    clf_RFC = RandomForestClassifier(n_estimators=4)

    score_SVC = clf_SVC.fit(train_set, train_label).score(valid_set, valid_label)
    score_NBR = clf_NBR.fit(train_set, train_label).score(valid_set, valid_label)
    score_RFC = clf_RFC.fit(train_set, train_label).score(valid_set, valid_label)

    print("SVC:", score_SVC)
    print("NBR:", score_NBR)
    print("RFC:", score_RFC)


# language_client = language.Client()
# train_google(language_client)
