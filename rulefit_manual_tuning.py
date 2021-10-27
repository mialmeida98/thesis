import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree
from sklearn.tree import _tree
import itertools
import sys
sys.setrecursionlimit(1000)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV



### A. IMPORTING DATASETS ###
#A.1. Import additional features (weather data)
weather = pd.read_csv("~/Desktop/Modelos Tese/LPPT.txt")
weather_time = weather.iloc[30840:63349,]

#Pre-Processing weather data
new_weather = pd.concat((weather_time['valid'], weather_time['tmpf']),axis = 1)
new_weather= pd.concat((new_weather, weather_time['relh']), axis = 1)
new_weather['tmpf_celsius'] = (new_weather['tmpf'] -32)/1.8
new_weather = new_weather.drop('tmpf', axis = 1)
new_weather['Day'] = new_weather['valid'].str[0:10]
new_weather = new_weather.groupby(by=['Day']).mean()
new_weather = new_weather.reset_index()


#A.2. Import prediction label (waiting times)
sns = pd.read_csv("~/Desktop/Modelos Tese/sns_df.csv")

#Pre-Processing waiting times
sns = sns.drop('Unnamed: 0', axis = 1)

#Choose hospital (Santa Maria) and emergency level
sns_hospital = sns[sns.Hospital.eq(218)]
sns_hospital = sns_hospital[sns_hospital.Emergency_Stage.eq(2)]


### B. OVERALL PRE-PROCESSING ###
#B.1. Group data by day
sns_hospital['Day'] = sns_hospital['Acquisition_Time'].str[0:10]
sns_hospital_grouped = sns_hospital.groupby(by=['Day']).mean()
day = pd.DataFrame(sns_hospital['Day'])
date = day.drop_duplicates()
people = sns_hospital_grouped.iloc[:, 2:3]
date = date.reset_index()
date = date.iloc[:,1:2]
people = people.reset_index()
people = people.iloc[:,1:2]
df_prophet = pd.concat((date, people), axis = 1)


#B.2. Preparing the dataframe
cov_final = new_weather.merge(df_prophet, left_on='Day', right_on='Day')
cov_final = cov_final.drop('Day', axis = 1)
cov_final = cov_final.fillna(cov_final.mean())
cov_final.to_csv('cov_final.csv')
X = cov_final.iloc[:,:-1]
y = cov_final['Waiting_Time']


#B.3. Split in Train, Validation and Test for Tuning
#Train
X_train = X[:112]
y_train = y[:112]
#Validation
X_val = X[112:122]
y_val = y[112:122]
#Test
X_test = X[122:]
y_test = y[122:]


### C. GRADIENT BOOSTING MACHINE ###
#C.1. Tuning
lr_list = [0.1, 0.25, 0.5, 0.75, 1, 1.1, 1.2, 1.5]
max_list = [3,4,5]
estimator_list = [1,10,100,500,1000]
min_mae = 100
for learning_rate in lr_list:
    for n_estimators in estimator_list:
        for max_depth in max_list:
            gb_clf = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_features=2, max_depth=max_depth, random_state=123)
            gb_clf.fit(X_train, y_train)
            
            print("Learning rate: ", learning_rate, 'Max Depth: ',max_depth, 'Estimators_',n_estimators)
            print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
            
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            scores = cross_val_score(gb_clf, X_val, y_val, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
            scores = np.absolute(scores)
            print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores))) 
            
            if np.mean(scores)<min_mae:
                min_mae = np.mean(scores)

#Choose best model based on minimum error (save the parameters for the implementation)
print('Minimum Error: ',min_mae)
