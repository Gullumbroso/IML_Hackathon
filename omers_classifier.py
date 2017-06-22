from google.cloud import language
from classifier import *
from sklearn.externals import joblib


def train_google(language_client):
    x, y = load_shuffled_data()

    res = []
    y_good = []
    for i in range(2100):
        if i%10 == 0:
            print(i)
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
    clf_NBR = neighbors.KNeighborsClassifier(5)
    clf_RFC = RandomForestClassifier(n_estimators=4)

    print(clf_SVC.fit(res[:1000], y_good[:1000]).score(res[1000:2000], y_good[1000:2000]))
    print(clf_NBR.fit(res[:1000], y_good[:1000]).score(res[1000:2000], y_good[1000:2000]))
    print(clf_RFC.fit(res[:1000], y_good[:1000]).score(res[1000:2000], y_good[1000:2000]))

    joblib.dump(clf_SVC, '/Users/omeralon/PycharmProjects/IML_Hackathon/dump/clf_SVC3.joblib.pkl', compress=9)
    joblib.dump(clf_NBR, '/Users/omeralon/PycharmProjects/IML_Hackathon/dump/clf_NBR3.joblib.pkl', compress=9)
    joblib.dump(clf_RFC, '/Users/omeralon/PycharmProjects/IML_Hackathon/dump/clf_RFC3.joblib.pkl', compress=9)




def predict_google(x):
    clf_SVC = joblib.load('/Users/omeralon/PycharmProjects/IML_Hackathon/dump/clf_SVC3.joblib.pkl')
    clf_NBR = joblib.load('/Users/omeralon/PycharmProjects/IML_Hackathon/dump/clf_NBR3.joblib.pkl')
    clf_RFC = joblib.load('/Users/omeralon/PycharmProjects/IML_Hackathon/dump/clf_RFC3.joblib.pkl')

    res = []
    for i in range(len(x)):
        document = language_client.document_from_text(x[i])
        sentiment = document.analyze_sentiment().sentiment
        vec = [sentiment.score, sentiment.magnitude]
        res.append(vec)

    return clf_SVC.predict(res), clf_NBR.predict(res), clf_RFC.predict(res)



if __name__ == "__main__":
    language_client = language.Client()
    train_google(language_client)


#x, y = load_dataset()
# print (y)

# print(y[:100])
# a,b,c = predict_google(x[:1000])
#
# print(sum(a))
# print(sum(b))
# print(sum(c))