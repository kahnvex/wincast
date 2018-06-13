import argparse as ap
import numpy as np
import pandas as pd
import os
import pkg_resources

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from itertools import chain
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.utils import shuffle
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

    def train(self):
        if self.args.indir:
            return self.read()

        self.scaler = preprocessing.StandardScaler()

        X = pd.read_csv(self.args.playdata)
        y = X.loc[:, 'y']
        X = self.get_features(X)
        X, y = shuffle(X, y)

        self.clf = LogisticRegressionCV()
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.15)

        X_train = self.scaler.fit_transform(X_train, y_train)
        X_test = self.scaler.transform(X_test, y_test)
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
        indir = pkg_resources.resource_filename('wincast', self.args.indir)
        self.clf = joblib.load(os.path.normpath(
            os.path.join(indir, 'wincast.clf.pkl')))
        self.scaler = joblib.load(os.path.normpath(
            os.path.join(indir, 'wincast.scaler.pkl')))

    def write(self):
        joblib.dump(self.clf, os.path.normpath(
            os.path.join(self.args.outdir, 'wincast.clf.pkl')))
        joblib.dump(self.scaler, os.path.normpath(
            os.path.join(self.args.outdir, 'wincast.scaler.pkl')))


if __name__ == '__main__':
    Trainer().train()
