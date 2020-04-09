# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:56:25 2020

@author: MY PC
"""

import pandas as pd
import matplotlib.pyplot as mt
import numpy as np

dataset = pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2:3].values

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)

#Fitting regression model
from sklearn.svm import SVR
reg = SVR(kernel ='rbf')
reg.fit(x,y)

#prediction
y_pred=sc_y.inverse_transform(reg.predict(sc_x.transform(np.array([[6.5]]))))

#visuallization
mt.scatter(x,y)
mt.plot(x,reg.predict(x))
mt.show()