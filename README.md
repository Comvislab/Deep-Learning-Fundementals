# Deep-Learning-Fundementals

In this repository, ComVIS Lab Team is going to share fundemental functions and fundemental algorithms used in Deep Learning Models.

In Deep Learning there are Activation Functions used for obtaining output of a single neuron.

Those functions can be "linear" or "non-linear". In Deep Learning "non-linear" activation functions are preferred.
Because with non-linear functions, "learning" can be done. For "linear" activation functions, the system behaves as
a linear model and its not suitable for learning complex data.

But we have to not forget that, for DL (Deep Learning) activation functions should be differentiable (I mean, we can
take gradient/take derivative of the function).

In the view of above explanations, we are giving well known activation functions here;
1. ReLU Activation: (act_fun_ReLU.py) In ReLU activation, the input is distorted when it below the zero (having 
  negative sign). This distortion makes the ReLU, non-linear one.
        ReLU = max (x,0)
        
2. Parameterized ReLU (pReLU) Activation: There is a problem related with classic ReLU function which is related 
with making all negative values to 0. This is known as Dying ReLU problem. To avoid this problem and also avoid from
pure linearity, for negative values, small linearity is applied with some scale constant.
        pReLU = max (x,0) + alpa * min (x,0)


This document is going to be expanded in time. 

With my pleasure.

Muhammed Cinsdikici
ComVIS Lab Team (Leader L0)
