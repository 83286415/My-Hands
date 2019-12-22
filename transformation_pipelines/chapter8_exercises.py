# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import warnings
import time

# Chapter import
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.pipeline import Pipeline
from sklearn.manifold import MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# Where to save the figures
WORKING_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
ROOT_PATH = os.path.join(WORKING_PATH, 'Hands on SK and TS\\')
CHAPTER_ID = "dimensionality_reduction"


def save_fig(fig_id, tight_layout=True):
    path = image_path(fig_id) + ".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)  # cannot save file if path doesn't exist


def image_path(fig_id):
    return os.path.join(ROOT_PATH, "images", CHAPTER_ID, fig_id)


def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = matplotlib.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=cmap(digit / 9))
    plt.axis("off")
    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)


if __name__ == '__main__':

    # Exercise 9:

    # Load MNIST
    mnist = fetch_mldata('MNIST original')
    X_train = mnist['data'][:60000]
    y_train = mnist['target'][:60000]

    X_test = mnist['data'][60000:]
    y_test = mnist['target'][60000:]

    # build model
    rnd_clf = RandomForestClassifier(random_state=42)

    # random forest training time
    t0 = time.time()
    rnd_clf.fit(X_train, y_train)
    t1 = time.time()
    print("Only Random Forest Training took {:.2f}s".format(t1 - t0))  # 5.13s

    # evaluate
    y_pred = rnd_clf.predict(X_test)
    print('Only Random Forest accuracy score: ', accuracy_score(y_test, y_pred))  # 0.9455

    # PCA before Random Forest
    pca = PCA(n_components=0.95)
    X_train_reduced = pca.fit_transform(X_train)

    # After PCA, random forest training time
    rnd_clf2 = RandomForestClassifier(random_state=42)
    t0 = time.time()
    rnd_clf2.fit(X_train_reduced, y_train)
    t1 = time.time()
    print("After PCA, Random Forest Training took {:.2f}s".format(t1 - t0))  # 12.00s much longer!
    '''Oh no! Training is actually more than twice slower now! How can that be? Well, as we saw in this chapter, 
    dimensionality reduction does not always lead to faster training time: it depends on the dataset, the model and the 
    training algorithm. See figure 8-6 (the manifold_decision_boundary_plot* plots above). If you try a softmax 
    classifier instead of a random forest classifier, you will find that training time is reduced by a factor of 3 when 
    using PCA. Actually, we will do this in a second, but first let's check the precision of the new random forest 
    classifier.'''

    # evaluate
    X_test_reduced = pca.transform(X_test)

    y_pred = rnd_clf2.predict(X_test_reduced)
    print('After PCA, Random Forest accuracy: ', accuracy_score(y_test, y_pred))  # 0.8908 a little drop as info lost
    '''It is common for performance to drop slightly when reducing dimensionality, because we do lose some useful signal
     in the process. However, the performance drop is rather severe in this case. So PCA really did not help: it slowed 
     down training and reduced performance. :('''

    # PCA doesn't help with training time and accuracy on Random Forest.
    # So try it on soft-max regression
    log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)  # 10 numbers so multi
    t0 = time.time()
    log_clf.fit(X_train, y_train)
    t1 = time.time()
    print("Only Logistic Regression Training took {:.2f}s".format(t1 - t0))  # 33.87s

    # evaluate
    y_pred = log_clf.predict(X_test)
    print('Only Logistic Regression accuracy score: ', accuracy_score(y_test, y_pred))  # 0.9255

    # After PCA, logistic regression training time
    log_clf2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
    t0 = time.time()
    log_clf2.fit(X_train_reduced, y_train)
    t1 = time.time()
    print("After PCA, Logistic Regression Training took {:.2f}s".format(t1 - t0))  # 11.12s much faster!

    # evaluate
    y_pred = log_clf2.predict(X_test_reduced)
    print('After PCA, Logistic Regression accuracy score: ', accuracy_score(y_test, y_pred))  # 0.9201 a little drop.

    # Exercise 10:

    # data set
    np.random.seed(42)

    m = 10000
    idx = np.random.permutation(60000)[:m]

    X = mnist['data'][idx]
    y = mnist['target'][idx]

    # build model
    tsne = TSNE(n_components=2, random_state=42)  # reduce MNIST data to 2 dimensions
    X_reduced = tsne.fit_transform(X)

    # plot
    plt.figure(figsize=(13, 10))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
    plt.axis('off')
    plt.colorbar()
    #plt.show()

    # focus on digital 2, 3 and 5
    plt.figure(figsize=(9, 9))
    cmap = matplotlib.cm.get_cmap("jet")
    for digit in (2, 3, 5):
        plt.scatter(X_reduced[y == digit, 0], X_reduced[y == digit, 1], c=cmap(digit / 9))
    plt.axis('off')

    # running t-SNE on 2(blue), 3, 5(green):
    idx = (y == 2) | (y == 3) | (y == 5)
    X_subset = X[idx]
    y_subset = y[idx]

    tsne_subset = TSNE(n_components=2, random_state=42)
    X_subset_reduced = tsne_subset.fit_transform(X_subset)

    # plot
    plt.figure(figsize=(9, 9))
    for digit in (2, 3, 5):
        plt.scatter(X_subset_reduced[y_subset == digit, 0], X_subset_reduced[y_subset == digit, 1], c=cmap(digit / 9))
    plt.axis('off')
    '''Much better, now the clusters have far less overlap. But some 3s are all over the place. Plus, there are two 
    distinct clusters of 2s, and also two distinct clusters of 5s. It would be nice if we could visualize a few digits 
    from each cluster, to understand why this is the case. Let's do that now.
Exercise: Alternatively, you can write colored digits at the location of each instance, or even plot scaled-down 
versions of the digit images themselves (if you plot all digits, the visualization will be too cluttered, so you should 
either draw a random sample or plot an instance only if no other instance has already been plotted at a close distance).
 You should get a nice visualization with well-separated clusters of digits.
Let's create a plot_digits() function that will draw a scatterplot (similar to the above scatterplots) plus write 
colored digits, with a minimum distance guaranteed between these digits. If the digit images are provided, they are 
plotted instead. This implementation was inspired from one of Scikit-Learn's excellent examples (plot_lle_digits, based 
on a different digit dataset).'''

    # plot with digital on scatter points
    plot_digits(X_reduced, y)

    # plot with digital images on scatter points
    plot_digits(X_reduced, y, images=X, figsize=(35, 25))

    # plot 2, 3, 5 with digital images on scatter points
    plot_digits(X_subset_reduced, y_subset, images=X_subset, figsize=(22, 22))

    # Using other dimensionality reduction algorithms such as PCA, LLE, or MDS and compare the resulting visualizations.
    # PCA time and visualizations
    t0 = time.time()
    X_pca_reduced = PCA(n_components=2, random_state=42).fit_transform(X)
    t1 = time.time()
    print("PCA took {:.1f}s.".format(t1 - t0))
    plot_digits(X_pca_reduced, y)

    # LLE time and visualizations
    t0 = time.time()
    X_lle_reduced = LocallyLinearEmbedding(n_components=2, random_state=42).fit_transform(X)
    t1 = time.time()
    print("LLE took {:.1f}s.".format(t1 - t0))
    plot_digits(X_lle_reduced, y)

    # PCA + LLE time and visualizations
    pca_lle = Pipeline([
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("lle", LocallyLinearEmbedding(n_components=2, random_state=42)),
    ])
    t0 = time.time()
    X_pca_lle_reduced = pca_lle.fit_transform(X)
    t1 = time.time()
    print("PCA+LLE took {:.1f}s.".format(t1 - t0))
    plot_digits(X_pca_lle_reduced, y)

    # MDS time and visualizations
    m = 2000
    t0 = time.time()
    X_mds_reduced = MDS(n_components=2, random_state=42).fit_transform(X[:m])
    t1 = time.time()
    print("MDS took {:.1f}s (on just 2,000 MNIST images instead of 10,000).".format(t1 - t0))
    plot_digits(X_mds_reduced, y[:m])

    # PCA + MDS time and visualizations
    pca_mds = Pipeline([
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("mds", MDS(n_components=2, random_state=42)),
    ])
    t0 = time.time()
    X_pca_mds_reduced = pca_mds.fit_transform(X[:2000])
    t1 = time.time()
    print("PCA+MDS took {:.1f}s (on 2,000 MNIST images).".format(t1 - t0))
    plot_digits(X_pca_mds_reduced, y[:2000])

    # LDA time and visualizations
    t0 = time.time()
    X_lda_reduced = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
    t1 = time.time()
    print("LDA took {:.1f}s.".format(t1 - t0))
    plot_digits(X_lda_reduced, y, figsize=(12, 12))
    # plt.show()

    # t-SNE time and visualizations
    t0 = time.time()
    X_tsne_reduced = TSNE(n_components=2, random_state=42).fit_transform(X)
    t1 = time.time()
    print("t-SNE took {:.1f}s.".format(t1 - t0))
    plot_digits(X_tsne_reduced, y)
    # plt.show()

    # PCA + t-SNE time and visualizations
    pca_tsne = Pipeline([
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("tsne", TSNE(n_components=2, random_state=42)),
    ])
    t0 = time.time()
    X_pca_tsne_reduced = pca_tsne.fit_transform(X)
    t1 = time.time()
    print("PCA+t-SNE took {:.1f}s.".format(t1 - t0))
    plot_digits(X_pca_tsne_reduced, y)
    # PCA roughly gave us a 25% speedup, without damaging the result. This is the best combination!
    plt.show()