#THIS CODE IS MODIFIED VERSION of 
# My STUDENT ISMET EMIR DEMIR(180315070) 
# 
# Objective		: Gold Price Prediction Using Simple Linear Regression
# CopyRight (C)	: Ismet Emir Demir & Muhammed Cinsdikici
# Date			: 30 March 2021, 13:20
# Code Repo		: ComVIS Lab 
# DataSet		: https://evds2.tcmb.gov.tr/index.php?/evds/serieMarket/collapse_25/5849/DataGroup/turkish/bie_mkaltytl/
#				  (Dataset is taken from Turkiye Cumhuriyeti Merkez Bankasi)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plt_line(w0, w1, points):
	abline_values = [w1 * i + w0 for i in points]

	# Plot the best fit line over the actual values
	plt.plot(x, abline_values, 'b')
	plt.title("Simple Linear Regression, w0:{} w1:{}".format(w0,w1))
	plt.xlabel("Normalized Day numbers inputs")
	plt.ylabel("Normalized Gold Prices")


data = pd.read_excel("EVDS.xlsx")

# Take the Prices as "y" (Real gold prices)
# Take the number of each day as "x" 

y = np.array(data['fiyat'].tolist());
x = np.array(range(1,len(y)+1));

# initialize w0 and w1 to random value. Lets take w0=0.43 w1=0.87
#derivative:  2.u.u' => (-2)*Sigma((y - w0 -w1x1)*x1)
# wi = wi - etha*derivative

w0=0.43
w1=0.87
#For lr(etha)=0.6    : If you normalize your inputs and outputs, 
#            =0.0001 : If you not normalize then use very very very small
lr=etha=0.6 
epochs = 10000
n= len(y)

# IF YOU NORMALIZE THE INPUT Learning Rate 0.6 is can be used 
# THEN summation is getting big example: gold price(240) the day (100) produces 24000 
# and saturation arises.
# Normalize with Z-Score: y= (y-np.mean(y))/np.std(y)
# DON'T forget to Normalize your TEST PATTERNS with THESE y_mean,y_std & x_mean,x_std
y_mean, y_std = np.mean(y), np.std(y)
y= (y-y_mean)/y_std
x_mean, x_std = np.mean(x), np.std(x)
x= (x-x_mean)/x_std

# IF YOU DON'T NORMALIZE THE INPUT AS WITH Z-SCORE's.. 
# YOU SHOULD TAKE LEARNING RATE very very very small number of 0.0001 or less than this.

#w0,w1,RSS = linear_regression(x, y, w0=0.3, w1=0.47, epochs=1000, learning_rate=0.0001)
#print(w0, w1, RSS)

for epoch in range(1,epochs):
	y_predicted  = w0 + w1*x;
	RSS	= sum((y-y_predicted)**2)
	d_RSS_dw0 = (-2/n) * sum(y-y_predicted)
	d_RSS_dw1 = (-2/n) * sum((y-y_predicted)*x)	
	w0 = w0 - (etha*d_RSS_dw0)
	w1 = w1 - (etha*d_RSS_dw1)
	print ("Epoch:{} w0:{} w1:{} Cost(RSS):{}".format(epoch,w0,w1,RSS));

plt.plot(x,y,"--")
plt_line(w0,w1,x)
plt.show()