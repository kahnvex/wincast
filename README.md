# wincast

A Real Time Win Forecasting System for the NFL

Note: This package is still under development.
For now, you can play around with it via the
command line.

## Usage

The model will predict whether or not the offense team will win.
An output of `1` means the model is forecasting a win for the offense
team. A `0` means the model is forecast a loss (or tie) for the
offense team.


```sh
$ pip install -r requirements.txt
$ python
>>> import numpy as np
>>> from wincast.train import Trainer
>>>
>>> model = Trainer()
>>> model.train()
>>> # Now you can make predictions. Input features are as follows:
>>> # (quarter, minute, second, points offense, points defense,
>>> #     t.o.l. offense, t.o.l. defense, down, yards to go,
>>> #     yards from own goal)
>>>
>>> # Here is an example of a call to predict, where the model
>>> # forecasts a win for the team on offense:
>>> model.predict([[4, 0, 5, 20, 7, 3, 2, 1, 2, 20]])
array([[1]], dtype=int32)

>>> # Get the probability of each class 0/1:
>>> model.predict_proba(np.array([[4, 0, 5, 20, 7, 3, 2, 1, 2, 20]]))
array([[ 0.00880867,  0.99119133]], dtype=float32)
```
