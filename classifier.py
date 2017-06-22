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


TRAIN_SET_SIZE = 0.8


class Classifier(object):

    def classify(self,X):
        """
        Recieves a list of m unclassified headlines, and predicts for each one which newspaper published it.
        :param X: A list of length m containing the headlines' texts (strings)
        :return: y_hat - a binary vector of length m
        """
        pass


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


def train_bag_of_words():

    # Get the data
    samples, labels = load_shuffled_data()

    # Transform the titles into vectors
    count_vect = CountVectorizer(ngram_range=(1, 3))
    X_train_counts = count_vect.fit_transform(samples)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Divide into train set and test set
    train_size = int(math.floor(TRAIN_SET_SIZE * len(samples)))
    x_train = X_train_tfidf[train_size:]
    y_train = labels[train_size:]
    x_test = X_train_tfidf[:train_size]
    y_test = labels[:train_size]

    # Train the classifier
    # clf = MultinomialNB().fit(x_train, y_train) # Score:  68
    # clf = LogisticRegression().fit(x_train, y_train) # Score: 64
    # clf = svm.SVC().fit(x_train, y_train)  # Score: 58

    sgd = SGDClassifier()
    linear_svc = svm.LinearSVC()

    sgd_score = repetitive_test(sgd, x_test, y_test, 1000)
    svc_score = repetitive_test(linear_svc, x_test, y_test, 1000)

    print("SGD: " + str(sgd_score))
    print("Linear SVC: " + str(svc_score))


train_bag_of_words()
