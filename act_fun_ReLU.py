############################################
# Instructor   : Prof.Dr.Muhammed Cinsdikici
# Lab		   : ComVIS Lab
# Date		   : 21 Mar 2021, 14:30
# Code Objctv  : ReLU Activation
# Copyright(C) : d2l.ai and ComVIS Lab
# Repository   : https://github.com/Comvislab/Deep-Learning-Fundementals
############################################

import tensorflow as tf
import matplotlib.pyplot as plt

#Produce x = [-8 -7.9 -7.8 ...0... 7.8 7.9 8] total 160 numbers
#x and y both are arrays of Tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)

#Using ReLU Function
y = tf.nn.relu(x)

# numpy() converts a tensor object into an numpy (ndarray object). 
# This implicitly means that the converted tensor will be 
# now processed on the CPU
plt.xlabel("x")
plt.ylabel("ReLU")
plt.plot(x.numpy(),y.numpy())

plt.show()





