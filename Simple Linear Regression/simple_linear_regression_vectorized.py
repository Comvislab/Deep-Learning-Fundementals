# Version Diff  : This is vectorized solution of Gradient Descent Approach for Simple Linear Regression
# Objective		: Gold Price Prediction Using Simple Linear Regression
# CopyRight (C)	: Muhammed Cinsdikici
# Date			: 01 Nov 2021, 13:00
# Code Repo		: ComVIS Lab 
                [https://github.com/Comvislab/Deep-Learning-Fundementals/tree/main/Simple%20Linear%20Regression]
# DataSet		: Synthetically produced data from formula, [y=05*x^3 + 0.3*x^2 + Random_Error]
#				  (Dataset is taken from Turkiye Cumhuriyeti Merkez Bankasi)

import numpy as np
import matplotlib.pyplot as plt

# Build Random Data Points (in the case belov, it is polynomial)
x_train = np.linspace(-1,1,101)
y_train = 0.5*x_train**3+0.3*x_train**2+np.random.rand(*x_train.shape)*0.3

# Our Model is Simple Linear Regression
def y_hat(w0,w1,x_input):
    y_hats = w0+np.multiply(x_input,w1)
    return (y_hats)

def RSS_der_w0(y,w0,w1,x_data):
    return -2*sum(y-y_hat(w0,w1,x_data))

def RSS_der_w1(y,w0,w1,x_data):
    residual=(y-y_hat(w0,w1,x_data))
    return -2*sum(residual*x_data)


w0=0.2
w1=0.5
eta=0.001

for i in range (0,50000):
    w0 = w0 - eta*RSS_der_w0(y_train,w0,w1,x_train)
    w1 = w1 - eta*RSS_der_w1(y_train,w0,w1,x_train)

plt.scatter(x_train,y_train)
y_hats= y_hat(w0,w1,x_train)
plt.plot(x_train,y_hats,"r")
plt.show()

