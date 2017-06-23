"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

             **  Dot Classifier  **

===================================================
"""
import sklearn.neighbors as nb


class DotClassifier:
    """
    DotClassifier is a classifier of headlines according to number of dots.
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
        :return: list of number of dots in each headline
        """
        res = []
        for headline in x:
            chars = [char for char in headline if char == "."]
            res.append([len(chars)])
        return res

    def predict(self, x):
        """
        :param x: list of number of dots in each headline
        :return: prediction o by of this classifier
        """
        samples = self.prepare_samples(x)
        return self.trained_model.predict(samples)


