############################################
# Instructor   : Prof.Dr.Muhammed Cinsdikici
# Lab		   : ComVIS Lab
# Date		   : 21 Mar 2021, 20:00
# Code Objctv  : Activation Functions and Gradients
# Copyright(C) : d2l.ai and ComVIS Lab
# Repository   : https://github.com/Comvislab/Deep-Learning-Fundementals
############################################

import tensorflow as tf
import matplotlib.pyplot as plt

def actReLU(a):
  return tf.nn.relu(a)

def actPReLU(a,par):
  return tf.nn.leaky_relu(a,par)

def actSigmoid(a):
  return tf.nn.sigmoid(a)
  
def actTanh(a):
  return tf.nn.tanh(a)

def sigmoid_gradient(realx):
	with tf.GradientTape() as t:
		realy = tf.nn.sigmoid(realx)
	return t.gradient(realy,realx)
  
def tanh_gradient(realx):
	with tf.GradientTape() as t:
		realy = tf.nn.tanh(realx)
	return t.gradient(realy,realx)


#Produce x = [-8 -7.9 -7.8 ...0... 7.8 7.9 8] total 160 numbers
#x and y both are arrays of Tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)

#Using ReLU Function
y1 = actReLU(x)
y2 = actPReLU(x,0.2)
y3 = actSigmoid(x)
grad_y3 = sigmoid_gradient(x)
y4 = actTanh(x)
grad_y4 = tanh_gradient(x)

# numpy() converts a tensor object into an numpy (ndarray object). 
# This implicitly means that the converted tensor will be 
# now processed on the CPU
plt.figure(1)
plt.xlabel("x")
plt.ylabel("Act Func")
plt.plot(x.numpy(),y1.numpy(),x.numpy(),y2.numpy(),x.numpy(),y3.numpy(),x.numpy(),y4.numpy())
plt.legend(["ReLU","pReLU","Sigmoid","TanH"])

plt.figure(2)
plt.xlabel("x")
plt.ylabel("Gradients of Act Func")
plt.plot(x.numpy(),grad_y3.numpy(),x.numpy(),grad_y4.numpy())
plt.legend(["GradSigmoid","GradTanH"])

plt.show()


