from classifier import *


def train_bag_of_words():

    # Get the data
    samples, labels = load_shuffled_data()

    # Transform the titles into vectors
    count_vect = CountVectorizer(ngram_range=(1, 3))
    X_train_counts = count_vect.fit_transform(samples)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Divide into train set and test set
    train_size = int(math.floor(TRAIN_SET_SIZE * len(samples)))
    x_train = X_train_tfidf[train_size:]
    y_train = labels[train_size:]
    x_test = X_train_tfidf[:train_size]
    y_test = labels[:train_size]

    # Train the classifier
    # clf = MultinomialNB().fit(x_train, y_train) # Score:  68
    # clf = LogisticRegression().fit(x_train, y_train) # Score: 64
    # clf = svm.SVC().fit(x_train, y_train)  # Score: 58

    sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    linear_svc = svm.LinearSVC()

    sgd_score = repetitive_test(sgd, x_test, y_test, 100)
    svc_score = repetitive_test(linear_svc, x_test, y_test, 100)

    print("SGD: " + str(sgd_score))
    print("Linear SVC: " + str(svc_score))


train_bag_of_words()