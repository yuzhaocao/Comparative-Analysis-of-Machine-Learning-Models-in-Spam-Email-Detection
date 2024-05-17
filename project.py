from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
import os

max_features = 5000


# Treat the entire message as a string, and removes the '\n' and '\r'
def load_one_file(filename):
    x = ""
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            x += line
    return x


# Iterate through all files in the specified folder and load the data
def load_files_from_dir(rootdir):
    x = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v = load_one_file(path)
            x.append(v)
    return x


# The folder where the data stored is located, stores normal emails inn ham and spam emails in spam
def load_all_files():
    ham = []
    spam = []
    # load from the first folder enron1
    for i in range(1, 2):
        path = "data/enron%d/ham/" % i
        print("Load %s" % path)
        ham += load_files_from_dir(path)
        path = "data/enron%d/spam/" % i
        print("Load %s" % path)
        spam += load_files_from_dir(path)
    return ham, spam


# Use bag-of-words modelling to vectorise email samples, ham with label 0, spam with label 1
def get_features_by_wordbag():
    ham, spam = load_all_files()
    x = ham + spam
    y = [0] * len(ham) + [1] * len(spam)
    vectorizer = CountVectorizer(
        decode_error='ignore',
        strip_accents='ascii',
        max_features=max_features,
        stop_words='english',
        max_df=1.0,
        min_df=1)
    x = vectorizer.fit_transform(x)
    x = x.toarray()
    return x, y


# Apply TF-IDF to refines the representation by weighing terms based on their significance in the document relative to the entire corpus
def get_features_by_wordbag_tfidf():
    ham, spam = load_all_files()
    x = ham + spam
    y = [0] * len(ham) + [1] * len(spam)
    vectorizer = CountVectorizer(binary=False,
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1)
    x = vectorizer.fit_transform(x)
    x = x.toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(x)
    x = tfidf.toarray()
    return x, y


# Building a Naive Bayes model
def nb_model(x_train, x_test, y_train, y_test):
    print("Naive Bayes Model")
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


# Building a SVM model
def svm_model(x_train, x_test, y_train, y_test):
    print("SVM Model")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


# Building a knn model
def knn_model(x_train, x_test, y_train, y_test):
    print("KNN Model")
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy Score: ", metrics.accuracy_score(y_test, y_pred))
    final_scores = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print("Precision =  %f Recall = %f F1 Score = %f" % final_scores[:3])
    print(metrics.confusion_matrix(y_test, y_pred))


# Building a LogisticRegression model
def lr_model(x_train, x_test, y_train, y_test):
    print("LR Model")
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy Score: ", metrics.accuracy_score(y_test, y_pred))
    final_scores = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print("Precision =  %f Recall = %f F1 Score = %f" % final_scores[:3])
    print(metrics.confusion_matrix(y_test, y_pred))


# Building a Deep Neural Network model
def dnn_model(x_train, x_test, y_train, y_test):
    print("DNN Model")

    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    print("With Bag of Words only")
    x, y = get_features_by_wordbag()
    # 60% train and 40% test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    nb_model(x_train, x_test, y_train, y_test)
    svm_model(x_train, x_test, y_train, y_test)
    knn_model(x_train, x_test, y_train, y_test)
    lr_model(x_train, x_test, y_train, y_test)
    dnn_model(x_train, x_test, y_train, y_test)

    print("Applying TF-IDF with Bag of Words")
    x, y = get_features_by_wordbag_tfidf()
    # 60% train and 40% test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    nb_model(x_train, x_test, y_train, y_test)
    svm_model(x_train, x_test, y_train, y_test)
    knn_model(x_train, x_test, y_train, y_test)
    lr_model(x_train, x_test, y_train, y_test)
    dnn_model(x_train, x_test, y_train, y_test)
