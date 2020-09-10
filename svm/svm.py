import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import load_data, test_n_trials, benchmark

range_cost = [2 ** i for i in range(-5, 8)]
range_gamma = [2 ** i for i in range(-8, 6)]
range_degrees = list(range(1, 4))


def make_title(classifier, components, n_trials=None):
    title = "SVM [kernel={}, C={}".format(classifier.kernel, classifier.C)
    if classifier.kernel == "poly":
        title += ", degree={}".format(classifier.degree)
    if classifier.kernel == "rbf":
        title += ", gamma={}".format(classifier.gamma)
    title += "] classification\n"
    if n_trials is not None:
        title += "({} trials), ".format(n_trials)
    if isinstance(components, tuple):
        title += "{} components: {}".format(components[0], components[1])
    elif isinstance(components, list):
        if len(components) == 22:
            title += "components: <all>"
        else:
            title += "components: {}".format(", ".join(components))
    else:
        title += "components: <all>"
    return title


def plot_decision_regions(clf, x_train, y_train, components):
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 2
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))
    cont = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    #
    #  PRINT RESULTS
    #
    plt.figure()
    plt.contourf(xx, yy, cont, alpha=0.8)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.xlabel(components[0])
    plt.ylabel(components[1])
    
    title = make_title(clf, components)
    
    plt.title("Decision regions of {}".format(title))
    # plt.savefig("images/rbf/{} using {} - regions.png".format(title, components))
    plt.show()


def benchmark_linear(components, n_trials=100, test_size=.10):
    x, y, features = load_data(format='numpy', components=components)
    
    accs = np.zeros((len(range_cost),))
    for i in range(len(range_cost)):
        c = range_cost[i]
        print("Testing C={}".format(c), file=sys.stderr)
        clf = SVC(C=c, kernel="linear")
        accs[i], _, _ = test_n_trials(clf, x, y, components, n_trials, test_size)
    
    print(accs)
    
    best_idx = np.argmax(accs)
    best_c = range_cost[best_idx]
    
    #
    #  RESULTS FOR BEST CLASSIFIER
    #
    best_classifier = SVC(C=best_c, kernel="linear")
    title = make_title(best_classifier, components, n_trials)
    print("Best results found with {}".format(title), file=sys.stderr)
    
    benchmark(title, best_classifier, components, n_trials, test_size, plot_decision_regions)


def benchmark_poly(components, n_trials=100, test_size=.10):
    x, y, features = load_data(format='numpy', components=components)
    
    accs = np.zeros((len(range_cost), len(range_degrees)))
    for i in range(len(range_cost)):
        c = range_cost[i]
        for j in range(len(range_degrees)):
            d = range_degrees[j]
            print("Testing C={}, degree={}".format(c, d), file=sys.stderr)
            clf = SVC(C=c, kernel="poly", degree=d)
            accs[i, j], _, _ = test_n_trials(clf, x, y, components, n_trials, test_size)
    
    print(accs)
    
    best_idx = np.argmax(accs)
    best_c = range_cost[best_idx // len(range_degrees)]
    best_d = range_degrees[best_idx % len(range_degrees)]
    
    #
    #  RESULTS FOR BEST CLASSIFIER
    #
    best_classifier = SVC(C=best_c, kernel="poly", degree=best_d)
    title = make_title(best_classifier, components, n_trials)
    print("Best results found with {}".format(title), file=sys.stderr)
    
    benchmark(title, best_classifier, components, n_trials, test_size, plot_decision_regions)


def benchmark_rbf(components, n_trials=100, test_size=.10):
    x, y, features = load_data(format='numpy', components=components)
    x = StandardScaler().fit_transform(x)
    
    accs = np.zeros((len(range_cost), len(range_gamma)))
    for i in range(len(range_cost)):
        c = range_cost[i]
        for j in range(len(range_gamma)):
            gamma = range_gamma[j]
            print("Testing C={}, gamma={}".format(c, gamma), file=sys.stderr)
            clf = SVC(C=c, kernel="rbf", gamma=gamma)
            accs[i, j], _, _ = test_n_trials(clf, x, y, components, n_trials, test_size)
    
    print(accs)
    
    best_idx = np.argmax(accs)
    best_c = range_cost[best_idx // len(range_gamma)]
    best_gamma = range_gamma[best_idx % len(range_gamma)]
    
    #
    #  RESULTS FOR BEST CLASSIFIER
    #
    best_classifier = SVC(C=best_c, kernel="rbf", gamma=best_gamma)
    title = make_title(best_classifier, components, n_trials)
    print("Best results found with {}".format(title), file=sys.stderr)
    
    benchmark(title, best_classifier, components, n_trials, test_size, plot_decision_regions)


if __name__ == '__main__':
    print("###", file=sys.stderr)
    print("###  SVM classification", file=sys.stderr)
    print("###\n", file=sys.stderr)
    
    n_trials = 10
    
    benchmark_linear(('LDA', 1), n_trials)
    benchmark_linear(('PCA', 1), n_trials)
    benchmark_linear(('PCA', 2), n_trials)
    benchmark_linear(('PCA', 3), n_trials)
    benchmark_linear(["odor", "spore-print-color"], n_trials)
    benchmark_linear(["odor", "spore-print-color", "gill-color"], n_trials)
    benchmark_linear(None, n_trials)

    benchmark_poly(('LDA', 1), n_trials)
    benchmark_poly(('PCA', 1), n_trials)
    benchmark_poly(('PCA', 2), n_trials)
    benchmark_poly(('PCA', 3), n_trials)
    benchmark_poly(["odor", "spore-print-color"], n_trials)
    benchmark_poly(["odor", "spore-print-color", "gill-color"], n_trials)
    benchmark_poly(None, n_trials)

    benchmark_rbf(('LDA', 1), n_trials)
    benchmark_rbf(('PCA', 1), n_trials)
    benchmark_rbf(('PCA', 2), n_trials)
    benchmark_rbf(('PCA', 3), n_trials)
    benchmark_rbf(["odor", "spore-print-color"], n_trials)
    benchmark_rbf(["odor", "spore-print-color", "gill-color"], n_trials)
    benchmark_rbf(None, n_trials)
