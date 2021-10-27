import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree
from sklearn.tree import _tree
import itertools
import sys
sys.setrecursionlimit(1000)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge


### A. DEFINE THE RULES (using output on rulefit_manual_rules) ###
def rule1(temp, relh):
 if (temp > 10.306) & (temp > 10.396) : return(1)
 else: return(0)
def rule2(temp, relh):
 if (temp <= 10.306) & (temp <= 10.25) : return(1)
 else: return(0)
def rule3(temp, relh):
 if (temp <= 10.306) & (temp > 10.25) : return(1)
 else: return(0)
def rule4(temp, relh):
 if (temp > 10.306) & (temp <= 10.396) : return(1)
 else: return(0)


### B. DEFINE THE DATAFRAME ###
#A.1. Import Dataframes
dataframe = pd.read_csv('dataframe.csv')
dataframe = dataframe.iloc[:,1:]
cov_final = pd.read_csv('cov_final.csv')

#A.2. Use output on rulefit_manual_dataframe to assemble final dataframe
dataframe['rule1']=cov_final[['tmpf_celsius','relh']].apply(lambda x: rule1(*x), axis=1)
dataframe['rule2']=cov_final[['tmpf_celsius','relh']].apply(lambda x: rule2(*x), axis=1)
dataframe['rule3']=cov_final[['tmpf_celsius','relh']].apply(lambda x: rule3(*x), axis=1)
dataframe['rule4']=cov_final[['tmpf_celsius','relh']].apply(lambda x: rule4(*x), axis=1)

#A.3. Define Train and Test Sets
data = dataframe.values
X, y = data[:, 1:], data[:,0]
X_train = X[368:492]
y_train = y[368:492]
X_test = X[492:]
y_test = y[492:]


### B. IMPLEMENT LASSO ###
model = Lasso(alpha = 0.5)
lasso1 = model.fit(X_train, y_train)
print('Coefficients on the LASSO: ',lasso1.coef_)

#B.1. Make Predictions
y_hat = model.predict(X_test)

#B.2. Evaluate the Model
print('MAE: ',mean_absolute_error(y_hat[:5], y_test[:5]))
print('MSE: ',mean_squared_error(y_hat[:5], y_test[:5]))
print('RMSE: ',np.sqrt(mean_squared_error(y_hat[:5], y_test[:5])))
