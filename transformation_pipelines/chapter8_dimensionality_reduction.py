# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import warnings

# Chapter import
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from deslib.des.knora_e import KNORAE

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


if __name__ == '__main__':

    # No code in Projection or Manifold Learning

    # Ignore useless warnings (see SciPy issue #5998)
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")