from google.cloud import language
from classifier import *
from sklearn.externals import joblib


def train_google(language_client):
    x, y = load_shuffled_data()

    res = []
    y_good = []
    for i in range(2100):
        try:
            document = language_client.document_from_text(x[i])
            sentiment = document.analyze_sentiment().sentiment
            vec = [sentiment.score, sentiment.magnitude]
            res.append(vec)
            y_good.append(y[i])
        except:
            continue
        i += 1

    clf_SVC = svm.SVC()

    print(clf_SVC.fit(res, y_good))

    joblib.dump(clf_SVC, '/Users/omeralon/PycharmProjects/IML_Hackathon/resources/clf_SVC.joblib.pkl', compress=9)



def predict_google(x):
    clf_SVC = joblib.load('/Users/omeralon/PycharmProjects/IML_Hackathon/resources/clf_SVC.joblib.pkl')

    res = []
    for i in range(len(x)):
        document = language_client.document_from_text(x[i])
        sentiment = document.analyze_sentiment().sentiment
        vec = [sentiment.score, sentiment.magnitude]
        res.append(vec)

    return clf_SVC.predict(res)


if __name__ == "__main__":
    client = language.Client()
    train_google(client)
