# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 19:17:14 2022

@author: Devashish
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Boston.csv")
df=pd.get_dummies(df,drop_first=True)

X=df.iloc[:,:-1]

y=df.iloc[:,-1]

X_train, X_test, y_train, y_test= train_test_split(X,y,train_size=0.2,random_state=2022)

#Create class of Linear Regression
class LR:

    def __init__(self ):
        pass

    # fit method function
    #arguments features (X) and labels (y)
    def fit(self, X,y):
        #school book method
        # adding X=1 for bias
        X_b = np.ones(X.shape[0]).reshape(-1,1) #initialize bias as 1
        
        
        X_w = X  #feature matrix
        
        X_wb = np.concatenate((X_b,X_w),axis = 1)   #concat bias weight and feature 
        
        # W = (X.T X)-1  (X.T y)
        
        #np.matmul(X_wb.T,X_wb)  ===> (X.T * X)
        
        #np.linalg.inv(np.matmul(X_wb.T,X_wb))  ===> (X.T *  X)-1
        
        #np.matmul(np.linalg.inv(np.matmul(X_wb.T,X_wb)), X_wb.T)  ===> (X.T X)-1 * X.T 
        
        #np.matmul(np.matmul(np.linalg.inv(np.matmul(X_wb.T,X_wb)), X_wb.T),y)  ===> (X.T X)-1  (X.T y)
        
        self.W_wb = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_wb.T,X_wb)), X_wb.T),y)
        
        #Bias is at 0th index
        self.intercept= self.W_wb[0]
        
        #Weight from 0th index onwards
        self.coeffs = self.W_wb[1:]
    
        return self
    
    #predict method function    
    def predict(self,X):
        X_b = np.ones(X.shape[0]).reshape(-1,1)
        X_w = X
        X_wb = np.concatenate((X_b,X_w),axis = 1)
        y_pred = np.matmul(X_wb,self.W_wb)
        
        return y_pred
        
    

#Instantiate LR class
fnlr = LR()


#Fit model on data
fnlr.fit(X_train,y_train)


#predict on test data
y_pred = fnlr.predict(X_test)


#x coefients
fnlr.coeffs 
"""
array([-9.96376947e-02,  1.88315720e-02,  2.76580395e-01,  2.50042248e+00,
       -1.03326202e+01,  6.14607416e+00,  3.03548150e-04, -5.59364382e-01,
        1.42940217e-01, -7.53768441e-03, -9.56563345e-01,  1.13106170e-02,
       -4.94078229e-01])
"""

#biases
fnlr.intercept #10.04518228086959

#Accuracy 
mean_squared_error(y_test, y_pred)  #27.796804416870874


