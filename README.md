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

3. Sigmoid Activation: It is used for classification. It gives the probability and also it is non-linear differentiable.
        Sigmoid = 1/1+e^z_in
        Gradient of Sigmoid:  d(Sigmoid)/dz_in  = e^(-z_in) / (1+ e^(-z_in))^2 = Sigmoid * (1-Sigmoid)
        
4. Tanh Activation: It is similar to sigmoid, except that it has center located at Point 0. 
        Tanh = (1 - e^(-2*z_in)) / (1 + e^(-2*z_in))
        Gradient of TanH:     d(Tanh)/dz_in = (1 - Tanh^2(z_in))

We are also given the gradient of the functions Sigmoid and TanH
This document is going to be expanded in time. 

With my pleasure.

Muhammed Cinsdikici
ComVIS Lab Team (Leader L0)
