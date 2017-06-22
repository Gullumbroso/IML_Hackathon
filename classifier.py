"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

            **  Headline Classifier  **

Auther(s): Gilad Lumbroso, Ady Kaiser, Omer Alon

===================================================
"""
import load_headlines
from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestClassifier


class Classifier(object):

    def classify(self,X):
        """
        Recieves a list of m unclassified headlines, and predicts for each one which newspaper published it.
        :param X: A list of length m containing the headlines' texts (strings)
        :return: y_hat - a binary vector of length m
        """


def lengh_array():
    x, y = load_headlines.load_dataset()
    train_size = int(0.8 * len(x))
    res = []
    for headline in x:
        res.append([len(headline)])
    train_set = res[:train_size]
    train_label = y[:train_size]
    valid_set = res[train_size:]
    valid_label = y[train_size:]
    clf_SVC = svm.SVC()
    clf_NBR = neighbors.KNeighborsClassifier(45, weights='distance')
    clf_RFC = RandomForestClassifier(n_estimators=10)
    score_SVC = clf_SVC.fit(train_set, train_label).score(valid_set, valid_label)
    score_NBR = clf_NBR.fit(train_set, train_label).score(valid_set, valid_label)
    score_RFC = clf_RFC.fit(train_set, train_label).score(valid_set, valid_label)
    print("SVC:", score_SVC)
    print("NBR:", score_NBR)
    print("RFC:", score_RFC)

lengh_array()
