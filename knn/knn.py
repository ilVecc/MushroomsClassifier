# -*- coding: utf-8 -*-
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from utils import load_data, test_n_trials, benchmark

range_k = list(range(1, 9))
range_p = list(range(3, 6))


def make_title(classifier, components, n_trials=None):
    title = "KNN [k={}, metric={}".format(classifier.n_neighbors, classifier.metric)
    if classifier.metric == "minkowski":
        title += ", p={}".format(classifier.p)
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


def benchmark_euclidean(components, n_trials=100, test_size=.10):
    x, y, features = load_data(format='numpy', components=components)
    x = StandardScaler().fit_transform(x)
    
    accs = np.zeros((len(range_k),))
    for i in range(len(range_k)):
        k = range_k[i]
        print("Testing k={}".format(k), file=sys.stderr)
        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
        accs[i], _, _ = test_n_trials(clf, x, y, components, n_trials, test_size)
    
    print(accs)
    
    best_idx = np.argmax(accs)
    best_k = range_k[best_idx]
    
    #
    #  RESULTS FOR BEST CLASSIFIER
    #
    best_classifier = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", n_jobs=-1)
    title = make_title(best_classifier, components, n_trials)
    print("Best results found with {}".format(title, best_k), file=sys.stderr)
    
    benchmark(title, best_classifier, components, n_trials, test_size)
    
    #
    # show accuracy comparison between different k values
    #
    best_classifier.n_neighbors = "?"
    plt.figure()
    plt.plot(np.array(range_k), accs)
    plt.title("Comparison of {}".format(make_title(best_classifier, components, n_trials)))
    plt.xlabel("n_neighbors")
    plt.ylabel("accuracy")
    # plt.savefig("images/knn_{}D_comparison.png".format(n_components))
    plt.show()


def benchmark_minkowski(components, n_trials=100, test_size=.10):
    x, y, features = load_data(format='numpy', components=components)
    x = StandardScaler().fit_transform(x)
    
    accs = np.zeros((len(range_k), len(range_p)))
    for i in range(len(range_k)):
        k = range_k[i]
        for j in range(len(range_p)):
            p = range_p[j]
            print("Testing k={}, p={}".format(k, p), file=sys.stderr)
            clf = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=p, n_jobs=-1)
            accs[i, j], _, _ = test_n_trials(clf, x, y, components, n_trials, test_size)
    
    print(accs)
    
    best_idx = np.argmax(accs)
    best_k = range_k[best_idx // len(range_p)]
    best_p = range_p[best_idx % len(range_p)]
    
    #
    #  RESULTS FOR BEST CLASSIFIER
    #
    best_classifier = KNeighborsClassifier(n_neighbors=best_k, metric="minkowski", p=best_p, n_jobs=-1)
    title = make_title(best_classifier, components, n_trials)
    print("Best results found with {}".format(title, best_k), file=sys.stderr)
    
    benchmark(title, best_classifier, components, n_trials, test_size)
    
    #
    # show accuracy comparison between different k values
    #
    best_classifier.n_neighbors = "?"
    plt.figure()
    plt.plot(np.array(range_k), accs)
    plt.title("Comparison of {}".format(make_title(best_classifier, components, n_trials)))
    plt.xlabel("n_neighbors")
    plt.ylabel("accuracy")
    plt.legend(["p={}".format(i) for i in range_p])
    # plt.savefig("images/{} - comparison.png".format(title))
    plt.show()


if __name__ == '__main__':
    print("###", file=sys.stderr)
    print("###  KNN classification", file=sys.stderr)
    print("###\n", file=sys.stderr)
    
    n_trials = 10
    
    benchmark_euclidean(("LDA", 1), n_trials)
    benchmark_euclidean(("PCA", 1), n_trials)
    benchmark_euclidean(("PCA", 2), n_trials)
    benchmark_euclidean(("PCA", 3), n_trials)
    benchmark_euclidean(["odor", "spore-print-color"], n_trials)  # manually selected features
    benchmark_euclidean(["odor", "spore-print-color", "gill-color"], n_trials)
    benchmark_euclidean(None, n_trials)  # all features
    
    benchmark_minkowski(("LDA", 1), n_trials)
    benchmark_minkowski(("PCA", 1), n_trials)
    benchmark_minkowski(("PCA", 2), n_trials)
    benchmark_minkowski(("PCA", 3), n_trials)
    benchmark_minkowski(["odor", "spore-print-color"], n_trials)
    benchmark_minkowski(["odor", "spore-print-color", "gill-color"], n_trials)
    benchmark_minkowski(None, n_trials)
