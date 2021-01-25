import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

OUTPUT_TEMPLATE = (
    'Bayesian classifier:    {bayes_train:.3f} {bayes_valid:.3f}\n'
    'kNN classifier:         {knn_train:.3f} {knn_valid:.3f}\n'
    'Rand forest classifier: {rf_train:.3f} {rf_valid:.3f}\n'
    'SVC classifier:         {svc_train:.3f} {svc_valid:.3f}\n'
     #'MLP classifier:         {mlp_train:.3f} {mlp_valid:.3f}\n'
    'DT classifier:          {dt_train:.3f} {dt_valid:.3f}\n'
    'VOT classifier:         {vot_train:.3f} {vot_valid:.3f}\n'
)


class classifier:
    def __init__(self, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y)
        bayes_model = make_pipeline(
            StandardScaler(),
            GaussianNB()
        )

        knn_model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors = 16)
        )

        rf_model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators = 1000 , max_depth = 8, min_samples_leaf = 10)
        )

        model_svc = make_pipeline(
            StandardScaler(),
            SVC(kernel = 'linear', C=0.1)
        )

        model_mlp = make_pipeline(
            StandardScaler(),
            MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4, 3), activation='logistic')
        )

        model_dt = make_pipeline(
            StandardScaler(),
            DecisionTreeClassifier(max_depth=7)
        )

        model_vot = make_pipeline(
            StandardScaler(),
            VotingClassifier([
                ('nb', GaussianNB()),
                ('knn', KNeighborsClassifier(15)),
                ('svm', SVC(kernel='linear', C=0.1)),
                ('tree1', DecisionTreeClassifier(max_depth=4)),
                ('tree2', DecisionTreeClassifier(min_samples_leaf=10)),
            ])
        )

        bayes_model.fit(X_train, y_train)
        knn_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        model_svc.fit(X_train, y_train)
        model_vot.fit(X_train, y_train)
        model_dt.fit(X_train, y_train)
        # model_mlp.fit(X_train, y_train)


        bayes_train = bayes_model.score(X_train, y_train)
        bayes_valid = bayes_model.score(X_valid, y_valid)
        knn_train = knn_model.score(X_train, y_train)
        knn_valid = knn_model.score(X_valid, y_valid)
        rf_train = rf_model.score(X_train, y_train)
        rf_valid = rf_model.score(X_valid, y_valid)


        print(OUTPUT_TEMPLATE.format(
            bayes_train=bayes_train,
            bayes_valid=bayes_valid,
            knn_train=knn_train,
            knn_valid=knn_valid,
            rf_train=rf_train,
            rf_valid=rf_valid,
            svc_train=model_svc.score(X_train, y_train),
            svc_valid=model_svc.score(X_valid, y_valid),
            vot_train=model_vot.score(X_train, y_train),
            vot_valid=model_vot.score(X_valid, y_valid),
            #mlp_train=model_mlp.score(X_train, y_train),
            #mlp_valid=model_mlp.score(X_valid, y_valid),
            dt_train=model_dt.score(X_train, y_train),
            dt_valid=model_dt.score(X_valid, y_valid),
        ))
