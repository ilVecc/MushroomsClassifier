import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm


def load_data(format=None, components=None):
    #
    #  DATA PREPROCESSING
    #
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "mushrooms.csv")).drop_duplicates()
    
    x = df.drop('class', axis=1)
    y = df['class']
    features = np.array(x.columns)
    
    # map strings to integers
    encoder_x = LabelEncoder()
    for col in x.columns:
        x[col] = encoder_x.fit_transform(x[col])
    encoder_y = LabelEncoder()
    y = encoder_y.fit_transform(y)
    
    if components is not None and not isinstance(components, tuple):
        x = x[components]
        features = components
    
    if format == 'numpy':
        x = np.array(x)
        y = np.array(y)
    
    return x, y, features


def reduce_data(x_train, y_train, x_test, components):
    if isinstance(components, tuple):
        if components[0] == 'PCA':
            reducer = PCA(n_components=components[1])
        elif components[0] == 'LDA':
            reducer = LinearDiscriminantAnalysis(n_components=1)
        else:
            raise ValueError("Cannot find that reduction!")
        reducer.fit(x_train, y_train)
        x_train = reducer.transform(x_train)
        x_test = reducer.transform(x_test)
    return x_train, x_test


def test_model(clf, x, y, components=None, test_size=.10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_test = reduce_data(x_train, y_train, x_test, components)
    
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    
    return accuracy_score(y_test, y_pred), \
           precision_score(y_test, y_pred, zero_division=0), \
           recall_score(y_test, y_pred, zero_division=0), \
           confusion_matrix(y_test, y_pred)


def test_n_trials(clf, x, y, components=None, n_trials=100, test_size=.10):
    stats = np.zeros((n_trials, 3))
    cms = np.zeros((2, 2))
    for i in tqdm(range(0, n_trials)):
        results = test_model(clf, x, y, components, test_size)
        stats[i, :] = np.array(results[:3])
        cms += results[3]
    cms = np.ceil(cms / n_trials).astype(int)
    return np.mean(stats[:, 0]), stats, cms


def benchmark(title, clf, components=None, n_trials=100, test_size=.10, plot_2d_fn=None):
    x, y, features = load_data(format='numpy', components=components)
    x = StandardScaler().fit_transform(x)
    
    _, stats, cm = test_n_trials(clf, x, y, components, n_trials, test_size)
    
    plot_stats(stats, "Stats of " + title)
    
    labels = ["edible", "poisonous"]
    plot_confusion_matrix(cm, "Avg. confusion matrix of " + title, labels)
    
    if plot_2d_fn is not None:
        if isinstance(components, list) and len(components) == 2:
            plot_2d_fn(clf, x, y, components)
        if isinstance(components, tuple) and components[1] == 2:
            plot_2d_fn(clf, x, y, ["PCA1", "PCA2"])


def plot_stats(results, title):
    max_results = np.max(results, axis=0) * 100
    avg_results = np.mean(results, axis=0) * 100
    min_results = np.min(results, axis=0) * 100
    
    stats = np.array([max_results, avg_results, min_results])
    
    cmap = "viridis"
    
    fig, ax = plt.subplots()
    image = ax.imshow(stats, interpolation='nearest', cmap=cmap)
    cmap_min, cmap_max = image.cmap(0), image.cmap(256)
    
    text = np.empty_like(stats, dtype=object)
    
    # print text with appropriate color depending on background
    thresh = (stats.max() + stats.min()) / 2.0
    
    for i in range(3):
        for j in range(3):
            color = cmap_max if stats[i, j] < thresh else cmap_min
            text_cm = "{:.2f}".format(stats[i, j])
            text[i, j] = ax.text(
                j, i, text_cm,
                ha="center", va="center",
                color=color)
    
    fig.colorbar(image, ax=ax)
    ax.set(xticks=[0, 1, 2],
           yticks=[0, 1, 2],
           xticklabels=["accuracy", "precision", "recall"],
           yticklabels=["best", "avg", "worst"],
           xlabel="Metric",
           ylabel="Performance")
    ax.set_ylim((2.5, -0.5))
    f = open(os.devnull, 'w')
    plt.setp(ax.get_xticklabels(), file=f)
    f.close()
    plt.title(title)
    plt.show()


def plot_confusion_matrix(cm, title, labels):
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
    plt.title(title)
    plt.show()


# noinspection PyUnresolvedReferences
def print_dataset(x_train, x_test, logger):
    logger.info("Train/Test ratio: {}/{}".format(100 - current_test_size * 100, current_test_size * 100))
    logger.info(" # Components: {}".format(x_train.shape[1]))
    logger.info("Total samples: {}".format(x_train.shape[0] + x_test.shape[0]))
    logger.info("Train samples: {}".format(x_train.shape[0]))
    logger.info(" Test samples: {}".format(x_test.shape[0]))
    logger.info("")
