from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from sklearn.model_selection import train_test_split 

pd.options.mode.copy_on_write = True

data = pd.DataFrame(load_data('Bikeshare'))
data['hr'] = data['hr'].astype('float64')
X = pd.DataFrame(data.drop(columns=['weathersit', 'mnth']))
X['intercept'] = 1
X.corr()

def Q3_a():
    with open('Q3_a_output.txt', 'w') as f:
        f.write(X.corr().to_string())
    
def Q3_b():
    fig, ax = plt.subplots()
    ax = X.plot.scatter(x='hr', y='bikers', ax=ax)
    ax.set_title('Number of Bikers vs Hr')
    plt.show()

def MSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    resid = y_true - y_pred
    resid2 = resid**2
    ssr = resid2.sum()
    mse = ssr / len(y_true)
    return mse

def Q3_c():
    model = sm.OLS(X['bikers'], X[['intercept', 'hr']])
    results = model.fit()
    print(summarize(results))
    y_pred = results.predict()
    print(f"MSE: {MSE(X['bikers'], y_pred)}")

def Q3_d():
    X['hr2'] = X['hr']**2
    model = sm.OLS(X['bikers'], X[['intercept', 'hr', 'hr2']])
    results = model.fit()
    print(summarize(results))
    y_pred = results.predict()
    print(f"MSE: {MSE(X['bikers'], y_pred)}")

def Q3_e():
    X['hr2'] = X['hr']**2
    X['hr x workingday'] = X['hr'] * X['workingday']
    model = sm.OLS(X['bikers'], X[['intercept', 'hr', 'hr2', 'workingday', 'hr x workingday']])
    results = model.fit()
    print(summarize(results))
    y_pred = results.predict()
    print(f"MSE: {MSE(X['bikers'], y_pred)}")
    
def Q3_f():
    X['hr2'] = X['hr']**2
    X['hr x workingday'] = X['hr'] * X['workingday']
    model = sm.OLS(X['bikers'], X[['intercept', 'hr', 'hr2', 'workingday', 'hr x workingday', 'temp']])
    results = model.fit()
    print(summarize(results))
    y_pred = results.predict()
    print(f"MSE: {MSE(X['bikers'], y_pred)}")
    
Q3_f()