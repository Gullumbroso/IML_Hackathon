"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

            **  Headline Classifier  **

Auther(s): Gilad Lumbroso, Ady Kaiser, Omer Alon, Or Dagan

===================================================
"""
import math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
import sklearn.svm as svm
import sklearn.neighbors as nb
from load_headlines import load_dataset
import length_classifier as lc
import polititians_classifier as pc
import bag_of_words_classifier as bowc
from sklearn.ensemble import RandomForestClassifier

TRAIN_SET_SIZE = 0.6
VALIDATION_SET_SIZE = 0.3
TEST_SET_SIZE = 0.1

CLASS_TRAIN_SET_SIZE = 0.7
CLASS_VALIDATION_SET_SIZE = 0.3


class Classifier(object):
    def classify(self, X):
        """
        Recieves a list of m unclassified headlines, and predicts for each one which newspaper published it.
        :param X: A list of length m containing the headlines' texts (strings)
        :return: y_hat - a binary vector of length m
        """
        # Get Data
        x, y = Classifier.load_shuffled_data()

        # Divide into train set and test set
        train_size = int(math.floor(CLASS_TRAIN_SET_SIZE * len(y)))
        x_train = x[:train_size]
        y_train = y[:train_size]
        x_valid = x[train_size:]
        y_valid = y[train_size:]

        # Get all the trained models
        # sc = sentiment_classifier(x_train, y_train)
        len_class = lc.LengthClassifier(x_train, y_train)
        bag_class = bowc.BagOfWordsClassifier(x_train, y_train)
        poli_class = pc.PolititiansClassifier(x_train, y_train)
        dot_class = pc.PolititiansClassifier(x_train, y_train)

        predictions = np.array([len_class.predict(x_valid), bag_class.predict(x_valid),
                                poli_class.predict(x_valid), dot_class.predict(x_valid)])

        pred_vecs = predictions.T

        linear_svc = svm.LinearSVC()
        linear_svc.fit(pred_vecs, y_valid)

        return linear_svc.predict(X)

    @staticmethod
    def load_clean_dataset():

        x, y = load_dataset(['resources/haaretz.csv', 'resources/israelhayom.csv'])
        clean_x = []
        clean_y = []
        for i in range(len(x)):
            if x[i] != "Haaretz Cartoon":
                clean_x.append(x[i])
                clean_y.append(y[i])

        return clean_x, clean_y

    @staticmethod
    def shuffle(x, y):
        pairs_list = list(zip(x, y))
        random.shuffle(pairs_list)
        new_x, new_y = zip(*pairs_list)
        return new_x, new_y

    @staticmethod
    def load_shuffled_data():
        x, y = Classifier.load_clean_dataset()
        return Classifier.shuffle(x, y)
