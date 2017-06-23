"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

          **  Bag of Words Classifier  **

===================================================
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sklearn.svm as svm


class BagOfWordsClassifier:
    """
    BagOfWordsClassifier is a classifier of headlines according to words uses.
    """

    def __init__(self, x, y):
        """
        trains the classifier model
        :param x: x train batch
        :param y: y train batch
        """
        self.count_vect = CountVectorizer(ngram_range=(1, 2))
        self.tfidf_transformer = TfidfTransformer()
        X_train_tfidf = self.prepare_samples(x)
        self.trained_model = svm.LinearSVC().fit(X_train_tfidf, y)

    def prepare_samples(self, x):
        """
        :param x: list of headlines 
        :return: 
        """
        X_train_counts = self.count_vect.fit_transform(x)
        return self.tfidf_transformer.fit_transform(X_train_counts)

    def predict(self, x):
        """
        :param x:  
        :return: prediction o by of this classifier
        """
        X_test_counts = self.count_vect.transform(x)
        X_test_tfidf = self.tfidf_transformer.transform(X_test_counts)
        return self.trained_model.predict(X_test_tfidf)
