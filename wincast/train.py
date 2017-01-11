import argparse as ap
import gc
import jsonpickle
import numpy as pd
import os
import pandas as pd
import sys

from itertools import chain

from jsonpickle.ext import numpy as jsonpickle_numpy

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class Trainer(object):
    def __init__(self, **kwargs):
        if bool(kwargs):
            self.args = self.get_args(self.to_args_list(kwargs))

        else:
            self.args = self.get_args()

    def get_args(self, args=None):
        parser = ap.ArgumentParser(description='Train wincast')

        parser.add_argument('--playdata', '-p', type=str, default='./data/Xy.csv')
        parser.add_argument('--nb_epoch', '-e', type=int, default=2)
        parser.add_argument('--verbose', '-v', default=False, action='store_true')

        parser.add_argument('--outdir', '-o', default=False)
        parser.add_argument('--indir', '-i', default=False)

        return parser.parse_args(args=args)


    def to_args_list(self, args):
        args = [['--%s' % key, value] for key, value in args.items()]

        return chain.from_iterable(args)


    def create_baseline(self):

        # create model
        model = Sequential()
        model.add(Dense(30, input_dim=10, init='normal', activation='relu'))
        model.add(Dense(30, init='normal', activation='relu'))
        model.add(Dense(1, init='normal', activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
            'accuracy', 'precision', 'recall'])

        return model


    def read(self):
        self.model = load_model(os.path.normpath(os.path.join(
            self.args.indir, 'wincast.model.h5')))
        self.scaler = joblib.load(os.path.normpath(os.path.join(
            self.args.indir, 'wincast.scaler.pkl')))


    def train(self):
        if self.args.indir:
            self.read()
            gc.collect()
            return

        estimators = []
        plays = pd.read_csv(self.args.playdata)

        X = plays.loc[:, [
            'qtr',
            'min',
            'sec',
            'ptso',
            'ptsd',
            'timo',
            'timd',
            'dwn',
            'ytg',
            'yfog'
        ]]

        y = plays.loc[:, 'y']

        self.scaler = preprocessing.StandardScaler()
        X = self.scaler.fit_transform(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        self.model = self.create_baseline()
        self.model.fit(X_train, y_train.as_matrix(),
                       nb_epoch=self.args.nb_epoch, verbose=self.args.verbose)

        if self.args.outdir:
            self.write()

        gc.collect()


    def evaluate(self):
        y_test_pred = self.model.predict(self.X_test)
        y_test_pred = y_test_pred > 0.5

        prf = precision_recall_fscore_support(self.y_test, y_test_pred)
        roc = roc_curve(self.y_test, y_test_pred)

        return prf, roc

    def _check_model(self):
        if not getattr(self, 'model', None):
            raise AttributeError('Model has not been trained, call train() before predicting')


    def predict(self, X, *args, **kwargs):
        self._check_model()

        return self.model.predict(self.scaler.transform(X), *args, **kwargs)


    def predict_proba(self, X, *args, **kwargs):
        self._check_model()

        return self.model.predict_proba(self.scaler.transform(X), **kwargs)


    def write(self):
        self.model.save(os.path.normpath(
            os.path.join(self.args.outdir, 'wincast.model.h5')))
        joblib.dump(self.scaler, os.path.normpath(
            os.path.join(self.args.outdir, 'wincast.scaler.pkl')))


if __name__ == '__main__':
    Trainer().train()
