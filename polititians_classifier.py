"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

            **  Politician Classifier  **

===================================================
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_data():
    """
    gets politicians data from data files
    :return: lists of politicians 
    """
    politicians_USA_1 = list(pd.read_csv('resources/USA.csv')['name'][:2865])
    politicians_israel = list(pd.read_csv('resources/KNESSET.csv')['name'][:2697])
    trump = list(pd.read_csv('resources/trump.csv')['name'])
    return politicians_USA_1, politicians_israel, trump


class PolititiansClassifier:
    """
    PolititiansClassifier is a classifier of headlines according to politicians appearances.
    """

    def __init__(self, x, y):
        """
        trains the classifier model
        :param x: x train batch
        :param y: y train batch
        """
        clf_RFC = RandomForestClassifier(n_estimators=4)
        res = self.prepare_samples(x)
        self.trained_model = clf_RFC.fit(res, y)

    def prepare_samples(self, x):
        """
        :param x: list of headlines 
        :return: list of number of politicians appears in each headline
        """
        politicians_USA_1, politicians_israel, trump = get_data()
        res = []
        for i in range(len(x)):
            counter = 0
            for j in range(len(politicians_israel)):
                if politicians_israel[j] in x[i]:
                    counter += 1
            for k in range(len(politicians_USA_1)):
                if politicians_USA_1[k] in x[i]:
                    counter += 1
            for u in range(len(trump)):
                if trump[u] in x[i]:
                    counter += 5
            res.append([counter])
        return res

    def predict(self, x):
        """
        :param x:  list of numbers
        :return: prediction o by of this classifier
        """
        samples = self.prepare_samples(x)
        return self.trained_model.predict(samples)
