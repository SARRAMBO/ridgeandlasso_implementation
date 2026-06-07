# Ridge ,Lasso,ElasticNet Implementation-(Hyperparameter tune the Linear Regression)
Ridge Regression(L2 Regularization)- In a linear regression algorithm, the best fit line generated training data is somewhat trying to touch all the training data points (overfitting) and has a low accuracy when subjected to test data in a bias-variance tradeoff (low-bias,high variance). To mitigate this we use Ridge regression which introduces another factor to the cost function of linear regression -
Cost function-
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \lambda \sum_{j=1}^{n} \theta_j^2$$

In other words,in Ridge Regression (also known as L2 regularization), the cost function is created by adding a penalty term to the standard Residual Sum of Squares (RSS). This penalty discourages the model's coefficients from becoming too large, which helps prevent overfitting.
It reduces the impact by reducing the coefficient of the least correlated feature in such a way that the best fit line's movement isnt much impact by that coefficient, which in turn ensures that the best fit line doesnt touch all the data points of the training data.

Best fit line(hypothetical behaviour)-
y=0.52+0.40x1+0.34x2+0.24x3 (before ridge)
y=0.52+0.30X1+0.28X2+0.10X3(after ridge regression)

Point to note is that,the least correlated feature doesnt get cancelled totally ,which happens in Lasso, as any of the slopes doesnt become zero
with increase in lambda (hyperparameter)

Lasso Regression(L1 regularization)- In Lasso Regression, the lowly correlated independent features wrt to the output/dependent variable get cancelled out resulting in feature selection.
Cost function-
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \lambda \sum_{j=1}^{n} |\theta_j|$$

Best fit line(hypothetical behaviour)-
y=0.52+0.40x1+0.44x2+0.60x3+0.10x4(before Lasso)
y=0.52+0.30x1+0.34x2+0.50x3+0.0x4(After Lasso , the x4 feature gets cancelled)


Elasticnet-(combination of Ridge and Lasso)-preferred in case of model overfitting and consisting of a high no of features
Cost function-
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \lambda_1 \sum_{j=1}^{n} |\theta_j| + \lambda_2 \sum_{j=1}^{n} \theta_j^2$$

The project thereby dicusses initial implementation of Linear Regression to Algerian Forest Fires Dataset and then how implementation of Ridge,Lasso and ElasticNet results in hyperparameter tuning the Regression model.It consists of basic EDA steps,feature engineering,data cleaning and model training which also includes cross validation techniques from the above three regression techniques(choosing best value for lambda after 100 iterations and 5 k-fold cross validations for each value of lambda).



