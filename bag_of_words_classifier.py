from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV  # This is the svm
import sklearn.svm as svm


class BagOfWordsClassifier:

    def __init__(self, x, y):

        self.count_vect = CountVectorizer(ngram_range=(1, 2))
        self.tfidf_transformer = TfidfTransformer()
        X_train_tfidf = self.prepare_samples(x)
        self.trained_model = svm.LinearSVC().fit(X_train_tfidf, y)

    def prepare_samples(self, x):
        X_train_counts = self.count_vect.fit_transform(x)
        return self.tfidf_transformer.fit_transform(X_train_counts)

    def predict(self, x):
        X_test_counts = self.count_vect.transform(x)
        X_test_tfidf = self.tfidf_transformer.transform(X_test_counts)
        return self.trained_model.predict(X_test_tfidf)
