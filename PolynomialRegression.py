# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:35:30 2020

@author: MY PC
"""

import pandas as pd
import matplotlib.pyplot as mt
import numpy as np

#importing dataset
dataset =pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#no need to encode data

#no need to split dataset

#linear regression on same dataset
from sklearn.linear_model import LinearRegression
lin_reg =LinearRegression()
lin_reg.fit(x,y)

#polynomial Regression on same dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

# visualization of linear models
#mt.scatter(x,y)
#mt.plot(x,lin_reg.predict(x))

# visualiztion of polynomial model
x_grid = np.arange(min(x),max(x),0.1)
x_grid= x_grid.reshape((len(x_grid),1))
mt.scatter(x,y)
mt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)))