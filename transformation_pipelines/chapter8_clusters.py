#  encoding=utf-8

import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pylab as pl
from operator import itemgetter
from collections import OrderedDict, Counter
from sklearn.datasets import make_moons


class DBScan (object):
    """
    the class inherits from object, encapsulate the  DBscan algorithm
    """
    def __init__(self, p, l_stauts):
        self.point = p  # X
        self.labels_stats = l_stauts  # y
        self.db = DBSCAN(eps=0.2, min_samples=10).fit(self.point)  # fit

    def draw(self):
        coreSamplesMask = np.zeros_like(self.db.labels_, dtype=bool)  # a array y with 0
        # db.labels_: the label list to show points belonging to which cluster. It's like y in normal data set.
        coreSamplesMask[self.db.core_sample_indices_] = True  # set core samples index to True
        labels = self.db.labels_  # Noisy samples are given the label -1
        nclusters = noise_reduction(labels)  # so need to noise reduction and return the count of clusters

        # model evaluation output: refer to cloud note and search 'clusters' for more details
        print('Estimated number of clusters: %d' % nclusters)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(self.labels_stats, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(self.labels_stats, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(self.labels_stats, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(self.labels_stats, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(self.labels_stats, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(self.point, labels))

        # plot points in colors and noise points in black
        unique_labels = set(labels)  # unique_labels: {0, 1, 2, 3, -1} a set, so no order and no repeated value
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = 'k'  # black for noise

            classMemberMask = (labels == k)  # np.array's operation:

            # plot each clusters' points
            xy = self.point[classMemberMask & coreSamplesMask]  # &: intersection of two sets
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

            # plot noise points
            xy = self.point[classMemberMask & ~coreSamplesMask]  # ~: not
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=3)

        plt.title('Estimated number of clusters: %d' % nclusters)
        # plt.show()


def noise_reduction(labels):
    clusters = len(set(labels)) - (1 if -1 in labels else 0)  # -1 means noise in labels
    return clusters  # return the count of clusters


def standard_scaler(points):
    p = StandardScaler().fit_transform(points)
    return p


if __name__ == "__main__":

    # DBScan

    # data prepare
    centers = [[1, 1], [-1, -1], [-1, 1], [1, -1]]  # center points
    point, labelsTrue = make_blobs(n_samples=2000, centers=centers, cluster_std=0.4, random_state=0)
    # generate 2000 points around center points
    # return X and its y

    # data reprocess
    point = standard_scaler(point)  # Standardize features by removing the mean and scaling to unit variance

    # build model and fit
    db = DBScan(point, labelsTrue)

    # plot
    db.draw()

    # K-means

    # data set
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]
    n_clusters = 4  # define the count of clusters

    # build model and fit
    cls = KMeans(n_clusters).fit(X)
    print(cls.labels_)

    # plot
    plt.figure(figsize=(6, 6))
    markers = ['^', 'x', 'o', '*', '+']
    for i in range(n_clusters):
        members = cls.labels_ == i  # members is list of [True or False], if i==labels it's True else False
        plt.scatter(X[members, 0], X[members, 1], s=60, marker=markers[i], c='b', alpha=0.5)
    plt.title('K-means clusters')
    # plt.show()

    # Layers clusters

    # data set
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    groups = [idx for idx in range(len(X))]  # each point is a group

    # distance dict of each pair of points
    disP2P = {}
    for idx1, point1 in enumerate(X):
        for idx2, point2 in enumerate(X):
            if idx1 < idx2:
                distance = pow(abs(point1[0] - point2[0]), 2) + pow(abs(point1[1] - point2[1]), 2)
                disP2P[str(idx1) + "#" + str(idx2)] = distance

    # re-order the distance dict in ascent order
    disP2P = OrderedDict(sorted(disP2P.items(), key=itemgetter(1), reverse=True))

    # core code
    groupNum = len(groups)  # total count groups
    finalGroupNum = int(groupNum * 0.1)  # the count of groups without noise
    while groupNum > finalGroupNum:
        # pick up two nearest points and add them into one group
        twopoins, distance = disP2P.popitem()
        pointA = int(twopoins.split('#')[0])  # A and B is the nearest points in dict
        pointB = int(twopoins.split('#')[1])
        pointAGroup = groups[pointA]
        pointBGroup = groups[pointB]
        if pointAGroup != pointBGroup:  # it not the same point group
            for idx in range(len(groups)):
                if groups[idx] == pointBGroup:
                    groups[idx] = pointAGroup  # add group A to group B and the count group - 1
            groupNum -= 1

    # define the count of final-group, so other groups are noises
    wantGroupNum = 6
    finalGroup = Counter(groups).most_common(wantGroupNum)  # a dict with 10 most common groups in the groups list
    finalGroup = [onecount[0] for onecount in finalGroup]  # finalGroup: the group index list
    dropPoints = [X[idx] for idx in range(len(X)) if groups[idx] not in finalGroup]

    # plot the most 6 common groups with different marks
    group1 = [X[idx] for idx in range(len(X)) if groups[idx] == finalGroup[0]]
    group2 = [X[idx] for idx in range(len(X)) if groups[idx] == finalGroup[1]]
    group3 = [X[idx] for idx in range(len(X)) if groups[idx] == finalGroup[2]]
    group4 = [X[idx] for idx in range(len(X)) if groups[idx] == finalGroup[3]]
    group5 = [X[idx] for idx in range(len(X)) if groups[idx] == finalGroup[4]]
    group6 = [X[idx] for idx in range(len(X)) if groups[idx] == finalGroup[5]]

    plt.figure(figsize=(17, 14))
    pl.plot([eachpoint[0] for eachpoint in group1], [eachpoint[1] for eachpoint in group1], 'or')
    pl.plot([eachpoint[0] for eachpoint in group2], [eachpoint[1] for eachpoint in group2], '+y')
    pl.plot([eachpoint[0] for eachpoint in group3], [eachpoint[1] for eachpoint in group3], 'sg')
    pl.plot([eachpoint[0] for eachpoint in group4], [eachpoint[1] for eachpoint in group4], 'pb')
    pl.plot([eachpoint[0] for eachpoint in group5], [eachpoint[1] for eachpoint in group5], 'vc')
    pl.plot([eachpoint[0] for eachpoint in group6], [eachpoint[1] for eachpoint in group6], '^m')

    # plot noise in black
    pl.plot([eachpoint[0] for eachpoint in dropPoints], [eachpoint[1] for eachpoint in dropPoints], 'xk')
    pl.show()