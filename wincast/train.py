import gc
import numpy as pd
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier


def create_baseline():

    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=10, init='normal', activation='relu'))
    model.add(Dense(30, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def main(verbose=0):
    estimators = []
    plays = pd.read_csv('./data/Xy.csv')

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

    estimators.append(('standardize', preprocessing.StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=2, verbose=verbose)))

    pipeline = Pipeline(estimators)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    pipeline.fit(X_train, y_train.as_matrix())

    gc.collect()

    return pipeline


if __name__ == '__main__':
    main(verbose=1)
