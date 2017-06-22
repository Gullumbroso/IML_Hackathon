from classifier import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def length_array():
    x, y = load_shuffled_data()
    res = []
    for headline in x:
        res.append([len(headline)])

    clf_NBR = nb.KNeighborsClassifier(40)
    clf_NBR.fit(res, y)
    return clf_NBR


def politicians(x, y):
    politicians_USA_1 = list(pd.read_csv('USA.csv')['name'][:2865])
    # politicians_UK = list(pd.read_csv('UK.csv')['name'][:2052])
    # politicians_USA_2 = list(pd.read_csv('USA2.csv')['name'][:626])
    politicians_israel = list(pd.read_csv('KNESSET.csv')['name'][:2697])
    trump = list(pd.read_csv('trump.csv')['name'])
    res = []
    for i in range(len(x)):
        counter = 0
        for j in range(len(politicians_israel)):
            if politicians_israel[j] in x[i]:
                counter += 1
        for k in range(len(politicians_USA_1)):
            if politicians_USA_1[k] in x[i]:
                counter += 1
        # for h in range(len(politicians_USA_2)):
        #     if politicians_USA_2[h] in x[i]:
        #         counter += 1
        # for v in range(len(politicians_UK)):
        #     if politicians_UK[v] in x[i]:
        #         counter += 1
        for u in range(len(trump)):
            if trump[u] in x[i]:
                counter += 5
        res.append([counter])
    clf_RFC = RandomForestClassifier(n_estimators=4)
    clf_RFC.fit(x, y)
    return clf_RFC


def import_politic2():
    politicians_USA_1 = list(pd.read_csv('USA.csv')['name'][:2865])
    politicians_UK = list(pd.read_csv('UK.csv')['name'][:2052])
    politicians_USA_2 = list(pd.read_csv('USA2.csv')['name'][:626])
    politicians_israel = list(pd.read_csv('KNESSET.csv')['name'][:2697])
    trump = list(pd.read_csv('trump.csv')['name'])
    x, y = load_shuffled_data()
    res = []
    for i in range(len(x)):
        counter = 0
        for j in range(len(politicians_israel)):
            if politicians_israel[j] in x[i]:
                counter += 1
        for k in range(len(politicians_USA_1)):
            if politicians_USA_1[k] in x[i]:
                counter += 1
        # for h in range(len(politicians_USA_2)):
        #     if politicians_USA_2[h] in x[i]:
        #         counter += 1
        # for v in range(len(politicians_UK)):
        #     if politicians_UK[v] in x[i]:
        #         counter += 1
        for u in range(len(trump)):
            if trump[u] in x[i]:
                counter += 5
        res.append([counter])
    clf_RFC = RandomForestClassifier(n_estimators=4)

    score_RFC = repetitive_test(clf_RFC, res, y, 400)
    print("RFC:", score_RFC)

import_politic2()