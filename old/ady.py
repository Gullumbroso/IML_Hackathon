"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

            **  Headline Classifier  **

Auther(s): Gilad Lumbroso, Ady Kaiser, Omer Alon, Or Dagan

===================================================
"""
from classifier import *

def repetitive_test(clf, x, y, num_of_tests):
    # Divide into train set and test set
    train_size = int(math.floor(TRAIN_SET_SIZE * len(y)))
    x_train = x[:train_size]
    y_train = y[:train_size]
    x_test = x[train_size:]
    y_test = y[train_size:]

    cul_score = 0
    for i in range(num_of_tests):
        shuffle(x_train, y_train)
        cul_score += clf.fit(x_train, y_train).score(x_test, y_test)

    return cul_score / num_of_tests


def master_classifier():
    # Get Data
    x, y = load_shuffled_data()

    # Divide into train set and test set
    train_size = int(math.floor(TRAIN_SET_SIZE * len(y)))
    valid_size = int(math.floor((TRAIN_SET_SIZE + VALIDATION_SET_SIZE) * len(y)))
    x_train = x[:train_size]
    y_train = y[:train_size]
    x_valid = x[train_size:valid_size]
    y_valid = y[train_size:valid_size]
    x_test = x[valid_size:]
    y_test = y[valid_size:]

    # Get all the trained models
    # sc = sentiment_classifier(x_train, y_train)
    len_class = lc.LengthClassifier(x_train, y_train)
    print("Length model ready.")
    bag_class = bowc.BagOfWordsClassifier(x_train, y_train)
    print("Bag of words model ready.")
    poli_class = pc.PolititiansClassifier(x_train, y_train)
    print("Polititians model ready.")
    dot_class = pc.PolititiansClassifier(x_train, y_train)
    print("Dot model ready.")

    predictions = np.array(
        [len_class.predict(x_valid), bag_class.predict(x_valid), poli_class.predict(x_valid), dot_class.predict(x_valid)])

    pred_vecs = predictions.T

    sgd = SGDClassifier()
    linear_svc = svm.LinearSVC()
    neighbors = nb.KNeighborsClassifier(40, weights='distance', algorithm='kd_tree')
    forest = RandomForestClassifier(n_estimators=10)

    # learn grida
    sgd.fit(pred_vecs, y_valid)
    linear_svc.fit(pred_vecs, y_valid)
    neighbors.fit(pred_vecs, y_valid)
    forest.fit(pred_vecs, y_valid)

    len_prediction = len_class.predict(x_test)
    bag_prediction = bag_class.predict(x_test)
    poli_prediction = poli_class.predict(x_test)
    dot_prediction = dot_class.predict(x_test)

    test_vec = test_vec2 = np.array([len_prediction,
                                     bag_prediction,
                                     poli_prediction,
                                     dot_prediction]).T

    sum_vec = np.sum(test_vec2, axis=1)
    sum_vec[sum_vec >= 3] = 8
    sum_vec[sum_vec <= 1] = 0
    sum_vec[sum_vec == 8] = 1
    sum_vec[sum_vec == 2] = bag_prediction[sum_vec == 2]

    sum_score = np.mean(sum_vec == y_test)

    sgd_score = sgd.score(test_vec, y_test)
    svc_score = linear_svc.score(test_vec, y_test)
    neighbors_score = neighbors.score(test_vec, y_test)
    forest_score = forest.score(test_vec, y_test)

    # Majority from predictions
    sgd_result = sgd.predict(test_vec)
    linear_result = linear_svc.predict(test_vec)
    neighbors_result = neighbors.predict(test_vec)
    forest_result = forest.predict(test_vec)

    result_matrix = np.array([sum_vec, sgd_result, linear_result, neighbors_result, forest_result]).T
    res_vec = np.sum(result_matrix, axis=1)
    res_vec[res_vec > 2] = 8
    res_vec[res_vec <= 2] = 0
    res_vec[res_vec == 8] = 1
    res_score = np.mean(res_vec == y_test)

    # svc_score = linear_svc.fit(pred_vecs, y_valid).score(x_test, y_test)

    # sgd_score = repetitive_test(sgd, x_test, y, 100)
    # svc_score = repetitive_test(linear_svc, X_train_tfidf, y, 100)

    print("SUM: " + str(sum_score))
    print("SGD: " + str(sgd_score))
    print("Linear SVC: " + str(svc_score))
    print("Neighbors: " + str(neighbors_score))
    print("Forest: " + str(forest_score))
    print("Majority: " + str(res_score))

    return np.argmax([sum_score, sgd_score, svc_score, neighbors_score, forest_score, res_score])

if __name__ == '__main__':
    results = [0] * 6
    for i in range(10):
        winner = master_classifier()
        results[winner] += 1
        print(results)

