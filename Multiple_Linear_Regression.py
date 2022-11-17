# -*- coding: utf-8 -*-

# Multiple Linear Regression From Scratch for using gradient descent
 
#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



#Read the CSV file as pandas dataframe
df=pd.read_csv("Housing.csv")

#One hot encoding of binary features
df=pd.get_dummies(df,drop_first=True)

#Feature columns
X=df.iloc[:,1:]

#Label column
y=df.iloc[:,0]

#Train test split of data
X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.3,random_state=2022)


#Create Linear regression with gradient descent class
class LRGD:

    def __init__(self,eta = 0.01, n_iterations = 1000 ):
        self.eta = eta
        self.n_iterations = n_iterations

    # fit method function
    #arguments features (X) and labels (y)
    def fit(self, X,y):
        #full batch gradient descent method
        # adding X=1 for bias
        
        #X_b ==> bias value
        X_b = np.ones(X.shape[0]).reshape(-1,1) #Convert to single column
        
        #X_w ==> feature values
        X_w = X
        
        #X_wb = X_w + X_b
        X_wb = np.concatenate((X_b,X_w),axis = 1)
        
        #Initialize the weights randomly for gradient descent
        #If y is single column i.e. single target
        if len(y.shape)==1:
            self.W_wb = np.random.random((X_wb.shape[1],))
            
        #If y has more than 1 column i.e. multiple targets
        else:
            self.W_wb = np.random.random((X_wb.shape[1],y.shape[1]))
            
            
        #Train model for n_iterations (EPOCHS)
        for iteration in range(self.n_iterations):
            
            # MSE = (y_pred - y_actual)**2
            # y_pred = X_wb * W_wb

		#gradient = 2/m  * X * ( X*W - y)
            gradients = 2./X.shape[0] * np.matmul( X_wb.T,  (np.matmul  (X_wb, self.W_wb)   - y)  )
          
            # New weights = Old weight - eta * gradients
            self.W_wb= self.W_wb - self.eta * gradients
         
                           
        self.intercept= self.W_wb[0]
        self.coeffs = self.W_wb[1:]
        self.gradients = gradients
        return self

    #predict method function    
    def predict(self,X):
        X_b = np.ones(X.shape[0]).reshape(-1,1)
        X_w = X
        X_wb = np.concatenate((X_b,X_w),axis = 1)
        y_pred = np.matmul(X_wb,self.W_wb)
        
        return y_pred
        

#Instantiate LR class
fnlrgd = LRGD()

#Scaling data using StandardScaler from sklearn
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#Fit model on data
fnlrgd.fit(X_train,y_train)

#Scale test data
X_test = sc.transform(X_test)

#predict on test data
y_pred = fnlrgd.predict(X_test)


#x coefients
fnlrgd.coeffs

#biases
fnlrgd.intercept

#gradients
fnlrgd.gradients