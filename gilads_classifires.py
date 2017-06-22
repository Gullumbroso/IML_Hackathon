from classifier import *


TRAIN_SET_SIZE = 0.8


def bag_of_words_classifier(x, y):

    # Transform the titles into vectors
    count_vect = CountVectorizer(ngram_range=(1, 2))
    X_train_counts = count_vect.fit_transform(x)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Divide into train set and test set
    # train_size = int(math.floor(TRAIN_SET_SIZE * len(samples)))
    # x_train = X_train_tfidf[train_size:]
    # y_train = labels[train_size:]
    # x_test = X_train_tfidf[:train_size]
    # y_test = labels[:train_size]

    # Train the classifier
    # clf = MultinomialNB().fit(x_train, y_train) # Score:  68
    # clf = LogisticRegression().fit(x_train, y_train) # Score: 64
    # clf = svm.SVC().fit(x_train, y_train)  # Score: 58

    sgd = SGDClassifier()
    linear_svc = svm.LinearSVC()

    sgd_score = repetitive_test(sgd, X_train_tfidf, y, 100)
    svc_score = repetitive_test(linear_svc, X_train_tfidf, y, 100)

    print("SGD: " + str(sgd_score))
    print("Linear SVC: " + str(svc_score))