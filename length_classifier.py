"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

            **  Length Classifier  **

===================================================
"""
from sklearn.ensemble import RandomForestClassifier
import sklearn.neighbors as nb
from sklearn.svm import SVC


class LengthClassifier:
    """
    LengthClassifier is a classifier of headlines according to the headline's length.
    """

    def __init__(self, x, y):
        """
        trains the classifier model
        :param x: x train batch
        :param y: y train batch
        """
        clf_NBR = nb.KNeighborsClassifier(40)
        res = self.prepare_samples(x)
        self.trained_model = clf_NBR.fit(res, y)

    def prepare_samples(self, x):
        """
        :param x: list of headlines 
        :return: list of headline's lengths
        """
        res = []
        for headline in x:
            res.append([len(headline)])
        return res

    def predict(self, x):
        """
        :param x: list of headline's lengths
        :return: prediction o by of this classifier
        """
        samples = self.prepare_samples(x)
        return self.trained_model.predict(samples)