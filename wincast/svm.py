import argparse as ap
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split


def parse_args():
    parser = ap.ArgumentParser()

    parser.add_argument('--val-split', float, default=0.15)
    parser.add_argument('--playdata', type=str, default='data/Xy.csv')

    return parser.parse_args()


def train(args):
    Xy = pd.read_csv(args.playdata)
    svm.SCV()
    X_train, X_test, y_train, y_test = train_test_split(X, y)


if __name__ == '__main__':
    train(parse_args())
