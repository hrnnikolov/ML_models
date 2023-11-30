import math

import numpy as np
import pandas as pd
from math import isclose
from sklearn.linear_model import LinearRegression
def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002
    cost_previous = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr -= learning_rate * md
        b_curr -= learning_rate * bd
        print(f'm {m_curr}, b {b_curr}, iteration {i}, cost {cost}')
        if math.isclose(cost,cost_previous):#, abs_tol=0.000001):
            print('Loop stopped')
            break
        cost_previous = cost

    return m_curr, b_curr

def pred_sklearn():
    df = pd.read_csv('test_scores.csv')

    r = LinearRegression()
    r.fit(df[['math']],df['cs'])

    print(f'm_sk {r.coef_},b_sk {r.intercept_}')


df = pd.read_csv('test_scores.csv')

m, b = gradient_descent(df['math'], df['cs'])

print(f'm_final {m}, b_final {b}')

pred_sklearn()