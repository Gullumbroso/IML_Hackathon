"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

            **  Sentiment Classifier  **

===================================================
"""
from google.cloud import language
from classifier import *
from sklearn.externals import joblib

language_client = language.Client()


def train_google():
    """
    Trains a SVC Model over the given headlines sets, features sentiments value of the text, using Google language.
    """
    x, y = load_shuffled_data()

    res = []
    y_good = []
    for i in range(len(x)):
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

    clf_SVC.fit(res, y_good)

    joblib.dump(clf_SVC, '/Users/omeralon/PycharmProjects/IML_Hackathon/resources/clf_SVC.joblib.pkl', compress=9)



def predict_google(x):
    """
    :param x: list of headlines
    :return: predictions of newspaper according to the model learn in "train_google()"
    """
    clf_SVC = joblib.load('/Users/omeralon/PycharmProjects/IML_Hackathon/resources/clf_SVC.joblib.pkl')

    res = []
    for i in range(len(x)):
        document = language_client.document_from_text(x[i])
        sentiment = document.analyze_sentiment().sentiment
        vec = [sentiment.score, sentiment.magnitude]
        res.append(vec)

    return clf_SVC.predict(res)


if __name__ == "__main__":
    train_google()
