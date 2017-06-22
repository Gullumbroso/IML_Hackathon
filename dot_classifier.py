from sklearn.ensemble import RandomForestClassifier
import sklearn.neighbors as nb


class DotClassifier:

    def __init__(self, x, y):

        clf_NBR = nb.KNeighborsClassifier(40)
        res = self.prepare_samples(x)
        self.trained_model = clf_NBR.fit(res, y)

    def prepare_samples(self, x):
        res = []
        for headline in x:
            chars = [char for char in headline if char == "."]
            res.append([len(chars)])
        return res

    def predict(self, x):
        samples = self.prepare_samples(x)
        return self.trained_model.predict(samples)


