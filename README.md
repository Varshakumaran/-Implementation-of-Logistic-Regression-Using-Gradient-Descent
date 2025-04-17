# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the dataset.

2.Define X and Y array.

3.Define a function for costFunction,cost and gradient.

4.Define a function to plot the decision boundary.

5.Define a function to predict the Regression value

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VARSHA K
RegisterNumber:  212223220122
*/
```
```
import pandas as pd
import numpy as np
```
```
dataset=pd.read_csv('Placement_Data.csv')
dataset
```

## Output:
![image](https://github.com/user-attachments/assets/edf28c53-e60e-489c-ab67-5266dc5cfb87)

```
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
```
```
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')    
dataset["status"]=dataset["status"].astype('category') 
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/35e2d850-6609-4c82-b1d0-edc372b75792)

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes   
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/afe21e3e-e87c-4c13-8afd-7c48c50ae401)

```
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/3713a629-90f6-43e6-8d02-5511783a9b99)
```
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred = np.where(h>= 0.5,1,0)
    return y_pred
y_pred = predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/9ed69d65-8b16-4beb-9778-9b3038c44a40)

```
print(y_pred)
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/2db9c6e0-70b9-4208-a684-555142e81cb0)

```
print(Y)
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/4ac5ce45-0c7b-4628-b8a9-2c92ec78385b)

```
xnew= np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/27861fbf-a4ee-4715-a623-9e3a4e69a796)

```
xnew= np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/414697aa-e3c4-4660-8b6e-6e4fdbb5fc93)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

