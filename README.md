# CustomXGBoost
XGBoost is the state of the art machine learning algorithm. So to get a good understanding of what is going under the hood of the XGBoost I implemented it from scratch using only Numpy.  
While implementing the gradient boosting I realize that it's similar to gradient descent in neural networks, so rather than just doing simple gradient descent, I tried using ADAM and RMSProp Optimizers and to my surprise, the results improved compared to Simple XGBoost.  
The comparison between algorithms is given inside `Comparision.ipynb` and all the code for CustomXGBoost is given in `CustomXGBoost.py`

# Overview of the CustomXGBoost
CustomXGBoost has 4 main Classes  

- **XGBRegressor** - for regression problem
- **XGBRegressorAdam** - XGBoost regressor with ADAM Optimizer
- **XGBRegressorRMS** - XGBoost regressor with RMSProp Optimizer
- **XGBClassifier** - for multiclass classification problem

implementational example 
### XGBRegressor
```
custom = XGBRegressor() # same for  XGBRegressorAdam and XGBRegressorRMS
custom.fit(X_train, y_train, eval_set = (X_test, y_test))
y_pred = custom.predict(X_test.values)
```
### XGBClassifier
```
custom = XGBClassifier(n_classes=5) # n_classes is total diiferent labels of target 
custom.fit(X_train, y_train)
y_pred = custom.predict(X_test)
```

All the other default values for parameters are as given in XGBoost official Documentation. 
