import argparse as ap
import gc
import jsonpickle
import numpy as pd
import os
import pandas as pd
import sys

from itertools import chain
from jsonpickle.ext import numpy as jsonpickle_numpy

from keras import callbacks
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization, Dense, Dropout, PReLU
from keras.wrappers.scikit_learn import KerasClassifier

from tabulate import tabulate

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Trainer(object):
    def __init__(self, **kwargs):
        if bool(kwargs):
            self.args = self.get_args(self.to_args_list(kwargs))

        else:
            self.args = self.get_args()

    def get_args(self, args=None):
        playdata = 'data/Xy.csv'

        parser = ap.ArgumentParser(description='Train wincast')

        parser.add_argument('--playdata', '-p', type=str, default=playdata)
        parser.add_argument('--nb-epoch', '-e', type=int, default=2)
        parser.add_argument('--verbose', '-v', default=False, action='store_true')
        parser.add_argument('--batch-size', '-b', type=int, default=128)
        parser.add_argument('--outdir', '-o', default=False)
        parser.add_argument('--indir', '-i', default=False)
        parser.add_argument('--monitor', '-m', action='store_true', default=False)
        parser.add_argument('--evaluate', action='store_true', default=False)
        parser.add_argument('--format', default='table', choices=('table', 'csv'))
        parser.add_argument('--headers', action='store_true', default=False)
        parser.add_argument('--board', action='store_true', default=False)

        return parser.parse_args(args=args)


    def to_args_list(self, args):
        args = [['--%s' % key, value] for key, value in args.items()]

        return chain.from_iterable(args)


    def create_baseline(self):

        # create model
        model = Sequential()

        model.add(Dense(32, input_dim=13, init='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.5))

        model.add(Dense(16, init='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.5))

        model.add(Dense(8, init='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.5))

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
            self.read()
            gc.collect()
            return

        self.scaler = preprocessing.StandardScaler()

        X = pd.read_csv(self.args.playdata) 
        X_train = X.loc[~X['seas'].isin([2001, 2008, 2016])] 
        X_test = X.loc[X['seas'].isin([2001, 2008, 2016])]

        y_train = X_train.loc[:, 'y']
        y_test = X_test.loc[:, 'y']
        
        X_train = self.get_features(X_train)
        X_test = self.get_features(X_test) 

        
        X_train = self.scaler.fit_transform(X_train, y_train)
        X_test = self.scaler.transform(X_test)

        
        X_train, y_train = shuffle(X_train, y_train)

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        self.model = self.create_baseline()
        
        train_callbacks = []

        if self.args.evaluate:
            train_callbacks.append(callbacks.CSVLogger('evaluate/training.csv'))
        
        if self.args.board:
            tb_cb = callbacks.TensorBoard(
                log_dir='evaluate/board/', histogram_freq=0,
                write_graph=True, write_images=False)
            train_callbacks.append(tb_cb)


        self.model.fit(X_train, y_train.as_matrix(),
                       batch_size=self.args.batch_size,
                       nb_epoch=self.args.nb_epoch,
                       verbose=self.args.verbose,
                       callbacks=train_callbacks)

        if self.args.evaluate:
            self.evaluate(X_test, y_test)

        if self.args.outdir:
            self.write()

        gc.collect()

    def evaluate(self, X, y):
        evaluation = self.model.evaluate(X, y.as_matrix())

        with open('evaluate/cv.csv', 'w+') as cv:
            cv.write(','.join(self.model.metrics_names))
            cv.write(os.linesep)
            cv.write(','.join(map(str, evaluation)))


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
