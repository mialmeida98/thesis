import pandas as pd
import numpy as np
import itertools
import sys
sys.setrecursionlimit(1000)

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


### B. CREATE DATAFRAME TO USE WITH LASSO, OWL or ElasticNet ###
dataframe=0
cov_final = pd.read_csv('cov_final.csv')
dataframe = pd.concat((cov_final['Waiting_Time'],cov_final['relh']),axis=1)
dataframe = pd.concat((dataframe, cov_final['tmpf_celsius']), axis=1)
dataframe.to_csv('dataframe.csv')

#B.1. Print code to generate final dataframe
for i in range(1,5): #insert maximum number of rules here
    print("dataframe['rule"+str(i)+"']=cov_final[['tmpf_celsius','relh']].apply(lambda x: rule"+str(i)+"(*x), axis=1)")

### EXAMPLE OF EXPECTED OUTPUT ###
#dataframe['rule1']=cov_final[['tmpf_celsius','relh']].apply(lambda x: rule1(*x), axis=1)
#dataframe['rule2']=cov_final[['tmpf_celsius','relh']].apply(lambda x: rule2(*x), axis=1)
#dataframe['rule3']=cov_final[['tmpf_celsius','relh']].apply(lambda x: rule3(*x), axis=1)
#dataframe['rule4']=cov_final[['tmpf_celsius','relh']].apply(lambda x: rule4(*x), axis=1)
