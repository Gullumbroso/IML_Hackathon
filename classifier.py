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
    x_train = x[train_size:]
    y_train = y[train_size:]
    x_test = x[:train_size]
    y_test = y[:train_size]

    cul_score = 0
    for i in range(num_of_tests):
        shuffle(x_train, y_train)
        cul_score += clf.fit(x_train, y_train).score(x_test, y_test)

    return cul_score / num_of_tests

