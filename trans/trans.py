# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import load_data


def manual_fisher_reduction():
    x, y, features = load_data(format='numpy')
    
    # standardization
    c1 = x[y == 0]  # poisonous
    c2 = x[y == 1]  # edible
    
    mu1 = np.mean(c1, axis=0)
    mu2 = np.mean(c2, axis=0)
    
    c1 = c1 - mu1
    c2 = c2 - mu2
    
    S1 = np.empty((len(features), len(features)))
    for i in range(c1.shape[0]):
        S1 += np.outer(c1[i, :], c1[i, :])
    
    S2 = np.empty((len(features), len(features)))
    for i in range(c2.shape[0]):
        S2 += np.outer(c2[i, :], c2[i, :])
    
    SW = S1 + S2
    
    w = np.linalg.pinv(SW).dot(mu1 - mu2)
    
    x = x.dot(w)
    
    plt.figure()
    plt.hist(
        x[y == 0],
        bins=100,
        histtype="barstacked",
        color="blue",
        alpha=0.9
    )
    plt.hist(
        x[y == 1],
        bins=100,
        histtype="barstacked",
        color="red",
        alpha=0.9
    )
    plt.legend(["edible", "poisonous"])
    plt.title("Fisher transformation")
    # plt.savefig("fisher_reduction.png")
    plt.show()


def manual_PCA_reduction(n_components=3):
    x, y, features = load_data(format='numpy')
    
    # standardization
    mu = np.mean(x, axis=0)
    x = x.astype(int) - mu
    
    # calculate eig
    cov = np.cov(x, rowvar=False)
    lambdas, U = np.linalg.eig(cov)
    ord_lambdas_idx = np.argsort(lambdas)[::-1]
    ord_lambdas_idx = ord_lambdas_idx[:n_components]
    
    U = U[:, ord_lambdas_idx]
    x = np.dot(x, U)
    
    features = features[ord_lambdas_idx]
    print("Selected features: {}".format(", ".join(features)))
    
    if len(features) >= 2:
        plt.figure()
        plt.scatter(np.array(x[:, 0]), np.array(x[:, 1]), c=y, cmap="coolwarm")
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title("PCA transformation\n (components: {})".format(", ".join(features[:2])))
        # plt.savefig("pca_{}_reduction_2d.png".format(list(features)))
        plt.show()
    
    if len(features) >= 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(np.array(x[:, 0]), np.array(x[:, 1]), np.array(x[:, 2]), c=y, cmap="coolwarm")
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        ax.text2D(0.05, 0.9, "PCA transformation\n (components: {})".format(", ".join(features)),
                  transform=ax.transAxes)
        # plt.savefig("pca_{}_reduction_3d.png".format(list(features)))
        plt.show()


def plot_reduction():
    x, y, features = load_data()
    
    x_feature = "cap-shape"  # "population"
    y_feature = "cap-surface"  # "spore-print-color"
    z_feature = "cap-color"
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(
        np.array(x[x_feature]),
        np.array(x[y_feature]),
        np.array(x[z_feature]),
        # np.ones((x.shape[0],)),
        c=y)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_zlabel(z_feature)
    # plt.savefig("images/components_reduction.png")
    plt.show()


if __name__ == '__main__':
    # plot_reduction()
    manual_PCA_reduction(3)
    manual_fisher_reduction()
