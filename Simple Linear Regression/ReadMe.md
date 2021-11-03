<h3>Simple Linear Regression with example Python coding for Gold Prices Prediction</h3>

In order to understand, we have to remember Gradient Descent;
\begin{gradient_formula}
w_j= w_j - \eta \sum_i (\text{y}^{(i)} - \text{hat{y}}^{(i)})x_{j}^{(i)}
\end{gradient_formula}

where w_j is the model coefficient, \eta is learning rate and (\text{y}^{(i)} - \text{hat{y}}^{(i)})x_{j}^{(i)}
is the partial derivative of the Loss Function according to the w_j

Here there are two variants of Gradient Descent Algorithm to find Simple Linear Regression solution.<br>
**[simplelinearregression.py] is the Gradient Descent Algorithm used with <br>
              "for loops" -classical programming approach- for updating the weights of SLR. <br>
**[simple_linear_regression_vectorized.py] is the Gradient Descent Algorithm used vectorized weight update.<br>

The Second one is better way of implementing Gradient Descent.

