import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_data():
    politicians_USA_1 = list(pd.read_csv('resources/USA.csv')['name'][:2865])
    politicians_israel = list(pd.read_csv('resources/KNESSET.csv')['name'][:2697])
    trump = list(pd.read_csv('resources/trump.csv')['name'])
    return politicians_USA_1, politicians_israel, trump


class PolititiansClassifier:

    def __init__(self, x, y):
        clf_RFC = RandomForestClassifier(n_estimators=4)
        res = self.prepare_samples(x)
        self.trained_model = clf_RFC.fit(res, y)

    def prepare_samples(self, x):
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
        samples = self.prepare_samples(x)
        return self.trained_model.predict(samples)
