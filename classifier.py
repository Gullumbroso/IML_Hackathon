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


def shuffle_data(x, y):
    pairs_list = list(zip(x, y))
    random.shuffle(pairs_list)
    new_x, new_y = zip(*pairs_list)
    return new_x, new_y


def train_bag_of_words():
    # Get the data
    samples, labels = load_dataset()

    train_size = int(math.floor(TRAIN_SET_SIZE * len(samples)))
    x_train = samples[train_size:]
    y_train = labels[train_size:]
    x_test = samples[:train_size]
    y_test = labels[:train_size]

    # Transform the titles into vectors
    count_vect = CountVectorizer(ngram_range=(1, 3))
    X_train_counts = count_vect.fit_transform(x_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Train the classifier
    # clf = MultinomialNB().fit(X_train_tfidf, articles.target)
    # clf = SGDClassifier().fit(X_train_tfidf, articles.target)
    # clf = LogisticRegression().fit(X_train_tfidf, articles.target)
    clf = svm.SVC().fit(X_train_tfidf, y_train)
    # clf = svm.LinearSVC().fit(X_train_tfidf, y_train)
    score = clf.score(x_test, y_test)

    print(score)


train_bag_of_words()
