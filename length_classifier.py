from sklearn.ensemble import RandomForestClassifier
import sklearn.neighbors as nb
from sklearn.svm import SVC


class LengthClassifier:

    def __init__(self, x, y):

        clf_NBR = nb.KNeighborsClassifier(40)
        res = self.prepare_samples(x)
        self.trained_model = clf_NBR.fit(res, y)

    def prepare_samples(self, x):
        res = []
        for headline in x:
            res.append([len(headline)])
        return res

    def predict(self, x):
        samples = self.prepare_samples(x)
        return self.trained_model.predict(samples)