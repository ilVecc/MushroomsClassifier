import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

from utils import load_data, plot_stats, plot_confusion_matrix


def fisher(data_train, labels_train, data_test):
    # standardization
    c1 = data_train[labels_train == 0]  # poisonous
    c2 = data_train[labels_train == 1]  # edible
    
    mu1 = np.mean(c1, axis=0)
    mu2 = np.mean(c2, axis=0)
    
    c1 = c1 - mu1
    c2 = c2 - mu2
    
    features = data_train.shape[1]
    
    S1 = np.empty((features, features))
    for i in range(c1.shape[0]):
        S1 += np.outer(c1[i, :], c1[i, :])
    
    S2 = np.empty((features, features))
    for i in range(c2.shape[0]):
        S2 += np.outer(c2[i, :], c2[i, :])
    
    SW = S1 + S2
    
    w = np.linalg.pinv(SW).dot(mu1 - mu2)
    
    return data_train.dot(w)[:, np.newaxis], data_test.dot(w)[:, np.newaxis]


def bayes(x, y, test_size, use_fisher=False):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    
    if use_fisher:
        x_train, x_test = fisher(x_train, y_train, x_test)
    
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    
    return accuracy_score(y_test, y_pred), \
           precision_score(y_test, y_pred, zero_division=0), \
           recall_score(y_test, y_pred, zero_division=0), \
           confusion_matrix(y_test, y_pred)


def benchmark(use_fisher, n_trials=100, test_size=.10):
    x, y, features = load_data(format='numpy')
    
    stats = np.zeros((n_trials, 3))
    confm = np.zeros((2, 2))
    for i in tqdm(range(0, n_trials)):
        results = bayes(x, y, test_size, use_fisher=use_fisher)
        stats[i, :] = np.array(results[:3])
        confm += results[3]
    confm = np.ceil(confm / n_trials).astype(int)
    
    title = "Stats of Bayesian classification ({} trials)".format(n_trials)
    plot_stats(stats, title)
    
    labels = ["edible", "poisonous"]
    title = "Avg. confusion matrix of Bayesian classification ({} trials)".format(n_trials)
    plot_confusion_matrix(confm, title, labels)


if __name__ == '__main__':
    print("###")
    print("###  Bayesian classification")
    print("###")
    
    benchmark(use_fisher=False)
    benchmark(use_fisher=True)
