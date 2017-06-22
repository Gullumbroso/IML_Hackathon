from google.cloud import language
from classifier import *
from sklearn.externals import joblib

def train_google(language_client):
    x, y = load_shuffled_data()

    res = []
    i = 0
    for headline in x:
        print(i)
        document = language_client.document_from_text(headline)
        sentiment = document.analyze_sentiment().sentiment
        res.append([sentiment.score, sentiment.magnitude])
        i += 1

    clf_SVC = svm.SVC()
    clf_NBR = neighbors.KNeighborsClassifier(40)
    clf_RFC = RandomForestClassifier(n_estimators=4)

    score_SVC = repetitive_test(clf_SVC, res, y, 100)
    score_NBR = repetitive_test(clf_NBR, res, y, 100)
    score_RFC = repetitive_test(clf_RFC, res, y, 100)

    joblib.dump(clf_SVC, '/dump/clf_SVC.joblib.pkl', compress=9)
    joblib.dump(score_NBR, '/dump/clf_NBR.joblib.pkl', compress=9)
    joblib.dump(score_RFC, '/dump/clf_RFC.joblib.pkl', compress=9)


    print("SVC:", score_SVC)
    print("NBR:", score_NBR)
    print("RFC:", score_RFC)


language_client = language.Client()
train_google(language_client)
