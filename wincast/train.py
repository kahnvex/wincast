import argparse as ap
import numpy as np
import pandas as pd
import os

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from itertools import chain

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


class Trainer(object):
    def __init__(self, **kwargs):
        if bool(kwargs):
            self.args = self.get_args(self.to_args_list(kwargs))

        else:
            self.args = self.get_args()

    def get_args(self, args=None):
        playdata = 'data/Xy.csv'

        parser = ap.ArgumentParser(description='Train wincast')

        parser.add_argument('--search-iter', type=int, default=32)
        parser.add_argument('--playdata', '-p', type=str, default=playdata)
        parser.add_argument('--outdir', '-o', default=False)
        parser.add_argument('--indir', '-i', default=False)
        parser.add_argument('--validation-split', type=float, default=0.15)

        return parser.parse_args(args=args)

    def to_args_list(self, args):
        args = [['--%s' % key, value] for key, value in args.items()]

        return chain.from_iterable(args)

    def get_features(self, X):
        return X.loc[:, [
            'qtr',
            'min',
            'sec',
            'ptso',
            'ptsd',
            'timo',
            'timd',
            'dwn',
            'ytg',
            'yfog',
            'ou',
            'pts_s',
            'off_h',
        ]]

    def param_search(self, X, y):
        n_iter = self.args.search_iter
        param_dist = {
            'loss': ['log'],
            'n_iter': [n_iter],
            'alpha': 10.0**(-np.arange(4,5)),
            'penalty': ['elasticnet', 'l1', 'l2'],
            'l1_ratio': 0.2 * np.arange(0,5),
            'learning_rate': ['constant','optimal', 'invscaling'],
            'eta0': 0.02 * np.arange(0,4) + 0.01
        }

        n_iter_search = self.args.search_iter
        clf = SGDClassifier()
        search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                    n_iter=n_iter_search)
        search.fit(X, y)

        return search.best_params_

    def train(self):
        if self.args.indir:
            return self.read()

        self.scaler = preprocessing.StandardScaler()

        X = pd.read_csv(self.args.playdata)
        y = X.loc[:, 'y']
        X = self.get_features(X)
        X, y = shuffle(X, y)

        X = self.scaler.fit_transform(X, y)
        params = self.param_search(X, y)

        self.clf = SGDClassifier(**params)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.15)
        self.clf.fit(X_train, y_train)
        test_acc = self.clf.score(X_test, y_test)

        print('Test accuracy: %s' % test_acc)

        if self.args.outdir:
            self.write()

    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))

    def predict(self, X):
        return self.clf.transform(self.scaler.transform(X))

    def get_data(self):
        X = pd.read_csv(self.args.playdata)
        y = X.loc[:, 'y']
        X = self.get_features(X)
        X, y = shuffle(X, y)

        return X, y

    def get_pipeline(self):
        pipeline = Pipeline(steps=[
            ('scaler', self.scaler),
            ('clf', self.clf)
        ])

        return pipeline

    def read(self):
        self.clf = joblib.load(os.path.normpath(
            os.path.join(self.args.indir, 'wincast.clf.pkl')))
        self.scaler = joblib.load(os.path.normpath(
            os.path.join(self.args.indir, 'wincast.scaler.pkl')))

    def write(self):
        joblib.dump(self.clf, os.path.normpath(
            os.path.join(self.args.outdir, 'wincast.clf.pkl')))
        joblib.dump(self.scaler, os.path.normpath(
            os.path.join(self.args.outdir, 'wincast.scaler.pkl')))


if __name__ == '__main__':
    Trainer().train()
