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

data: pd.DataFrame = load_data('Bikeshare') # type: ignore

X = data['temp']
df = pd.DataFrame(X)
df['intercept'] = 1
X = df[['intercept','temp']]
y = data['bikers']
y = pd.DataFrame(y)
Train: pd.DataFrame = pd.merge_ordered(X,y,left_on=X.index,right_on=y.index).drop(columns=['key_0']) # type: ignore

model = sm.OLS(y, X)
results = model.fit()

def Q2_a():
    print(summarize(results))

def Q2_c():
    print("R^2: ", results.rsquared)

def Q2_e():
    print("min temp: ", X['temp'].min())

def Q2_g():
    fig, ax = plt.subplots()
    ax = Train.plot.scatter(x='temp', y='bikers', ax=ax)
    ax.set_title('Bikeshare Data: Number of Bikers vs Temperature')
    intercept, slope = results.params
    ax.axline((0, intercept), slope=slope, color="red", label="OLS fit")
    plt.show()
    
def Q2_h():
    fig, ax = plt.subplots()
    y_predicted = results.fittedvalues
    residuals = y['bikers'] - y_predicted
    ax.scatter(y_predicted, y['bikers'])
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Observed Values')
    ax.set_title('Predicted vs Observed Values')

    fig, ax = plt.subplots()
    ax.scatter(y_predicted, residuals)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Predicted vs Residuals')
    plt.show()
    
    
    
Q2_h()