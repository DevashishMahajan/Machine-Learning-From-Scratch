import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
#%matplotlib inline

Linear_data = pd.read_csv("LinearRegressionTest.csv") #LinearRegressionTest.csv   #Housing.csv
print(Linear_data)

X_train = Linear_data['lotsize'].to_numpy().reshape(-1,1)
y_train= Linear_data['price']

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)

print(X_train)
print(y_train)


plt.scatter(X_train,y_train)
plt.show()

def loss_function(m,b,X_train,y_train):
    Square_Error=0
    for i in range(len(y_train)):
        x=X_train[i]
        y=y_train[i]
        Square_Error += (y - (m * x + b))**2
    Mean_Square_Error = Square_Error/float(len(y_train))
    return Mean_Square_Error

m=1
b=0
print("MSE =",loss_function(m,b,X_train,y_train)) #MSE = 1243019.4210526317

m=2
b=0
print("MSE =",loss_function(m,b,X_train,y_train)) #MSE = 1215814.5263157894

m=5
b=0
print("MSE =",loss_function(m,b,X_train,y_train)) #MSE = 1136011.8421052631


#### Gradient descent #### 
def gradient_descent(X_train,y_train,m_now=1,b_now=0,alpha=0.001):
    m_gradient=0 
    b_gradient=0

    n=len(y_train)

    for i in range(n):
        x=X_train[i]
        y=y_train[i]

        m_gradient+= -(2)*x*(y- ((m_now *x) + b_now))
        b_gradient+= -(2) * (y - ((m_now *x) +b_now))

    print("m_now =",m_now," || ", "b_now =",b_now) 
    print("m_gradient=",m_gradient/n," || ","b_gradient=",b_gradient/n)
    #print("m",m,"b",b)
    m= m_now - ((m_gradient/n) *alpha)
    b= b_now - ((b_gradient/n) *alpha)
    print("m = ",m," || ","b = ",b)
    return m,b


m=1
b=0
ALPHA=0.1
EPOCHS=60
for i in range(EPOCHS):
    print()
    print("EPOCHS =",i, "  ||  ","ALPHA =",ALPHA)
    m,b = gradient_descent(X_train,y_train,m_now=m,b_now=b,alpha=ALPHA)
    

x_init=min(X_train)
y_init=min(y_train)
x_final=max(X_train)
y_final=max(y_train)
plt.scatter(X_train,y_train)
plt.plot([x_init,x_final],[y_init,y_final],color = 'r', linestyle = '-')
y_init_pred = x_init*m + b
y_final_pred = x_final*m +b
#Predicted line after applying linear regression using gradient descent
plt.plot([x_init,x_final],[y_init_pred,y_final_pred],color = 'y', linestyle = '--')
plt.show()

