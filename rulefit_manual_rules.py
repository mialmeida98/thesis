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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



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
#C.1. Implemented Best Model Based on Hyperparameters obtained in rulefit_manual_tuning
#Define Hyperparamters
estimators = 1
lr = 0.1
m_features = 2
m_depth = 2
#Implement Model
gb_clf2 = GradientBoostingRegressor(n_estimators=estimators, learning_rate=lr, max_features=m_features, max_depth=m_depth, random_state=123)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_test)
#Evaluate Model
mean_squared_error(predictions, y_test)
mean_absolute_error(predictions, y_test)


### D. EXTRACT RULES ###
def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += ' & '
            rule += str(p)
        rule += " : "
        if class_names is None:
            rule += "return("+str(1)+")"
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        #rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules


#D.1. Create list with the rules
rules = []
for i in range(estimators):
  rules.append(get_rules(gb_clf2.estimators_[i][0], ['relh','temp'], None))
#D.2. Print all the rules as functions to use in the next step
k = 1
for i in range(len(rules)):
  for j in range(len(rules[i])):
    print('def rule'+str(k)+'(temp, relh):')
    print(' '+rules[i][j])
    print(' '+'else: return(0)')
    k+=1

pd.DataFrame(rules).to_csv('rules.csv')

### EXAMPLE OF EXPECTED OUTPUT ###
#def rule1(temp, relh):
# if (temp > 10.306) & (temp > 10.396) : return(1)
# else: return(0)
#def rule2(temp, relh):
# if (temp <= 10.306) & (temp <= 10.25) : return(1)
# else: return(0)
#def rule3(temp, relh):
# if (temp <= 10.306) & (temp > 10.25) : return(1)
# else: return(0)
#def rule4(temp, relh):
# if (temp > 10.306) & (temp <= 10.396) : return(1)
# else: return(0)
