<h3>Simple Linear Regression with example Python coding for Gold Prices Prediction</h3>

In order to understand, we have to remember Gradient Descent;
\begin{gradient_formula}
w_j= w_j - \eta \sum_i (\text{y}^{(i)} - \text{hat{y}}^{(i)})x_{j}^{(i)}
\end{gradient_formula}

where \theta_i is the model coefficient, \alpha is learning rate and \frac{1}{n}\sum_{i=1}^n (h_\theta(x) - y_i)*x_j
is the partial derivative of the Loss Function according to the \theta_i

Here there are two variants of Gradient Descent Algorithm to find Simple Linear Regression solution.
The First one [simplelinearregression.py] is the Gradient Descent Algorithm used with 
              "for loops" -classical programming approach- for updating the weights of SLR.
The Second one [simple_linear_regression_vectorized.py] is the Gradient Descent Algorithm used vectorized weight update.

The Second one is better way of implementing Gradient Descent.

