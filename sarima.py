import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller



### A. IMPORTING DATASETS ###
#A.1. Import additional features (weather data)
weather = pd.read_csv("~/Desktop/Modelos Tese/LPPT.txt")
weather_time = weather.iloc[30840:63349,]

#Pre-Processing weather data
new_weather = pd.concat((weather_time['valid'], weather_time['tmpf']),axis = 1)
new_weather= pd.concat((new_weather, weather_time['relh']), axis = 1)
new_weather= pd.concat((new_weather, weather_time['sknt']), axis = 1)
new_weather= pd.concat((new_weather, weather_time['drct']), axis = 1)
new_weather= pd.concat((new_weather, weather_time['dwpf']), axis = 1)
new_weather['tmpf_celsius'] = (new_weather['tmpf'] -32)/1.8
new_weather = new_weather.drop('tmpf', axis = 1)
new_weather['dwpf_celsius'] = (new_weather['dwpf'] -32)/1.8
new_weather = new_weather.drop('dwpf', axis = 1)
new_weather['Day'] = new_weather['valid'].str[0:11]
new_weather = new_weather.groupby(by=['Day']).mean()


#A.2. Import prediction label (waiting times)
sns = pd.read_csv("~/Desktop/Modelos Tese/sns_df.csv")

#Pre-Processing waiting times
sns = sns.drop('Unnamed: 0', axis = 1)
new_sns = sns.drop('H_Name', axis = 1)
new_sns.head(n=2)

#Choose hospital (Santa Maria) and emergency level
sns_hospital = new_sns[new_sns.Hospital.eq(218)]
sns_hospital_1 = sns_hospital[sns_hospital.Emergency_Stage.eq(1)]




### B. OVERALL PRE-PROCESSING ###

#B.1. Group data by day
sns_hospital_1['Day'] = sns_hospital_1['Acquisition_Time'].str[0:11]
sns_hospital_1.head(2)
sns_hospital_day = sns_hospital_1.groupby(by=['Day']).mean()


#B.2. Additional features
#Weekdays
weekDaysMapping = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
weekDayInstance = pd.to_datetime(sns_day_1['Day'], infer_datetime_format=True)
weekdays = []
for i in range(len(sns_day_1)):
    weekdays.append(weekDaysMapping[weekDayInstance[i].weekday()])
weeknumbers = []
for i in range(len(sns_day_1)):
    weeknumbers.append(weekDayInstance[i].weekday())


#B.3. Preparing the dataframe
y = sns_day_1.set_index(['Day'])
y = y.drop('Hospital', axis = 1)
y = y.drop('Emergency_Stage', axis = 1)
y = y.drop('People_Waiting', axis = 1)
y_previ = y.iloc[:,0:1] #label
y_exog = new_weather #exogenous features


#B.3. Extra: standardize the data

#Original Data
mean_y = y['Waiting_Time'].mean()
std_y = y['Waiting_Time'].std()
y_zscored = y
y_zscored['Waiting_Time_Z'] = (y_zscored['Waiting_Time']-mean_y)/std_y
y_zscored = y_zscored.drop('Waiting_Time', axis = 1)

#Exogenous Features
mean_yexog_temp = y_exog['tmpf_celsius'].mean()
std_yexog_temp = y_exog['tmpf_celsius'].std()
mean_yexog_relh = y_exog['relh'].mean()
std_yexog_relh = y_exog['relh'].std()
mean_yexog_sknt = y_exog['sknt'].mean()
std_yexog_sknt = y_exog['sknt'].std()
mean_yexog_drct = y_exog['drct'].mean()
std_yexog_drct = y_exog['drct'].std()
mean_yexog_dwpf = y_exog['dwpf_celsius'].mean()
std_yexog_dwpf = y_exog['dwpf_celsius'].std()
y_zscored_exog = y_exog
y_zscored_exog['tmpf_Z'] = (y_zscored_exog['tmpf_celsius']-mean_yexog_temp)/std_yexog_temp
y_zscored_exog['relh_Z'] = (y_zscored_exog['relh']-mean_yexog_relh)/std_yexog_relh
y_zscored_exog['drct_Z'] = (y_zscored_exog['drct']-mean_yexog_drct)/std_yexog_drct
y_zscored_exog['sknt_Z'] = (y_zscored_exog['sknt']-mean_yexog_sknt)/std_yexog_sknt
y_zscored_exog['dwpf_Z'] = (y_zscored_exog['dwpf_celsius']-mean_yexog_dwpf)/std_yexog_dwpf
y_zscored_exog = y_zscored_exog.drop('tmpf_celsius', axis = 1)
y_zscored_exog = y_zscored_exog.drop('relh', axis = 1)
y_zscored_exog = y_zscored_exog.drop('sknt', axis = 1)
y_zscored_exog = y_zscored_exog.drop('drct', axis = 1)
y_zscored_exog = y_zscored_exog.drop('dwpf_celsius', axis = 1)

#Fill Nans with mean value 
y_zscored_exog = y_zscored_exog.fillna(y_zscored_exog.mean())


#Merge the original dataframe with the exogenous features dataframe and with the weekdays
y_zscored = y_zscored.reset_index()
y_zscored_exog = y_zscored_exog.reset_index()
covariates_final = y_zscored.merge(y_zscored_exog, left_on='Day', right_on='Day')
covariates_final = pd.concat((covariates_final,pd.DataFrame(weeknumbers)),axis=1)
covariates_final = covariates_final.rename(columns={0:'WeekNumber'})
covariates_final = covariates_final.set_index(['Day'])



### C. SARIMA IMPLEMENTATION ###

#C.1. Prepare the data
y_zscored = covariates_final.iloc[:,0:1]
y_zscored_exog = covariates_final.iloc[:,1:]

#C.2. Stationarity
ad_fuller_result = adfuller(y_zscored['Waiting_Time_Z'])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

#C.3. ACF
plt.figure(figsize =[20,20])
plot_acf(y_zscored, lags = 20)
plt.show()

#C.4. PACF
plot_pacf(y_zscored)
plt.show() #colocar p até 1


#C.5. Tuning Parameters
#Manually imput different seasonal orders and save the best one based on AIC and BIC
#The remaining parameters are tuned based on a grid search defined above
p = d = q = range(0, 2)
pdq = [(x[0], 0, x[2]) for x in list(itertools.product(p, d, q))]
seasonal_pdq = [(x[0], 0, x[2], 2) for x in list(itertools.product(p, d, q))] #specify the seasonal order here

aics = []
bics = []
orders = []
seasonal_orders = []
for i in pdq:
    for j in seasonal_pdq:
        model = sm.tsa.statespace.SARIMAX(y_zscored, exog = y_zscored_exog['2017':], order=i, seasonal_order=j)
        results = model.fit()
        print('order:',i,'seasonal order: ', j, 'AIC: ', results.aic)
        aics.append(results.aic)
        bics.append(results.bic)
        orders.append(i)
        seasonal_orders.append(j)

aic_results = pd.DataFrame(aics)
aic_results = aic_results.rename(columns = {0: 'AIC'})
bic_results = pd.DataFrame(bics)
bic_results = bic_results.rename(columns = {0: 'BIC'})
orders = pd.DataFrame(orders)
seasonal = pd.DataFrame(seasonal_orders)
out1 = pd.concat((aic_results, bic_results), axis = 1)
out2 = pd.concat((out1, orders), axis = 1)
out3 = pd.concat((out2, seasonal), axis = 1)
out3 = out3.drop_duplicates()
out3.sort_values(by = 'AIC').head(5)


#C.6. Fit the Best Model

#Model Summary
best_model = sm.tsa.statespace.SARIMAX(y_zscored, exog = y_zscored_exog['2017':],order=(1, 0, 1), seasonal_order=(0, 0, 0, 2)).fit()
print(best_model.summary())
#Model Diagnostics
best_model.plot_diagnostics(figsize=(15,12))
plt.show()


### D. EVALUATE THE MODEL BASED ON 5-FOLD CROSS-VALIDATION ###

#D.1. First Fold
#Prepare the data
y_z_train = y_zscored.iloc[:490,:]
y_z_train_exog = y_zscored_exog.iloc[:490,:]
exog_dim = np.array(y_zscored_exog.iloc[489:490,:])
exog_dim = np.reshape(exog_dim, (1, 6))
#Fit the model
best_model_train = sm.tsa.statespace.SARIMAX(y_z_train, exog = y_z_train_exog['2017':],order=(1, 0, 1), seasonal_order=(0, 0, 0, 2)).fit()
#Make predictions
y_z_train['arima_model'] = best_model_train.fittedvalues
forecast = best_model_train.predict(start=y_z_train.shape[0], end=y_z_train.shape[0], exog = exog_dim)
forecast = y_z_train['arima_model'].append(forecast)
#Remove the standardization
real_values = y_zscored.iloc[490:491,:]*std_y+mean_y
pred_values = pd.DataFrame(forecast.iloc[490:491])*std_y+mean_y
#MAE
mean_absolute_error(pred_values[0], real_values['Waiting_Time_Z'])
#MSE
mean_squared_error(pred_values[0], real_values['Waiting_Time_Z'])

#D.2.Second Fold
#Prepare the data
y_z_train = y_zscored.iloc[:491,:]
y_z_train_exog = y_zscored_exog.iloc[:491,:]
exog_dim = np.array(y_zscored_exog.iloc[490:491,:])
exog_dim = np.reshape(exog_dim, (1, 6))
#Fit the Model
best_model_train = sm.tsa.statespace.SARIMAX(y_z_train, exog = y_z_train_exog['2017':],order=(1, 0, 1), seasonal_order=(0, 0, 0, 2)).fit()
#Make predictions
y_z_train['arima_model'] = best_model_train.fittedvalues
forecast = best_model_train.predict(start=y_z_train.shape[0], end=y_z_train.shape[0], exog = exog_dim)
forecast = y_z_train['arima_model'].append(forecast)
#Remove standardization
real_values = y_zscored.iloc[491:492,:]*std_y+mean_y
pred_values = pd.DataFrame(forecast.iloc[491:492])*std_y+mean_y
#MAE
mean_absolute_error(pred_values[0], real_values['Waiting_Time_Z'])
#MSE
mean_squared_error(pred_values[0], real_values['Waiting_Time_Z'])


#D.3. Third Fold
#Prepare the data
y_z_train = y_zscored.iloc[:492,:]
y_z_train_exog = y_zscored_exog.iloc[:492,:]
exog_dim = np.array(y_zscored_exog.iloc[491:492,:])
exog_dim = np.reshape(exog_dim, (1, 6))
#Fit the model
best_model_train = sm.tsa.statespace.SARIMAX(y_z_train, exog = y_z_train_exog['2017':],order=(1, 0, 1), seasonal_order=(0, 0, 0, 2)).fit()
y_z_train['arima_model'] = best_model_train.fittedvalues
forecast = best_model_train.predict(start=y_z_train.shape[0], end=y_z_train.shape[0], exog = exog_dim)
forecast = y_z_train['arima_model'].append(forecast)
#Remove standardization
real_values = y_zscored.iloc[492:493,:]*std_y+mean_y
pred_values = pd.DataFrame(forecast.iloc[492:493])*std_y+mean_y
#MAE
mean_absolute_error(pred_values[0], real_values['Waiting_Time_Z'])
#MSE
mean_squared_error(pred_values[0], real_values['Waiting_Time_Z'])


#D.4. Fourth Fold
#Prepare the data
y_z_train = y_zscored.iloc[:493,:]
y_z_train_exog = y_zscored_exog.iloc[:493,:]
exog_dim = np.array(y_zscored_exog.iloc[492:493,:])
exog_dim = np.reshape(exog_dim, (1, 6))
#Fit the model
best_model_train = sm.tsa.statespace.SARIMAX(y_z_train, exog = y_z_train_exog['2017':],order=(1, 0, 1), seasonal_order=(0, 0, 0, 2)).fit()
y_z_train['arima_model'] = best_model_train.fittedvalues
forecast = best_model_train.predict(start=y_z_train.shape[0], end=y_z_train.shape[0], exog = exog_dim)
forecast = y_z_train['arima_model'].append(forecast)
#Remove standardization
real_values = y_zscored.iloc[493:494,:]*std_y+mean_y
pred_values = pd.DataFrame(forecast.iloc[493:494])*std_y+mean_y
#MAE
mean_absolute_error(pred_values[0], real_values['Waiting_Time_Z'])
#MSE
mean_squared_error(pred_values[0], real_values['Waiting_Time_Z'])


# ##### Fifth Fold
#Prepare the Data
y_z_train = y_zscored.iloc[:494,:]
y_z_train_exog = y_zscored_exog.iloc[:494,:]
exog_dim = np.array(y_zscored_exog.iloc[493:494,:])
exog_dim = np.reshape(exog_dim, (1, 6))
#Fit the model
best_model_train = sm.tsa.statespace.SARIMAX(y_z_train, exog = y_z_train_exog['2017':],order=(1, 0, 1), seasonal_order=(0, 0, 0, 2)).fit()
y_z_train['arima_model'] = best_model_train.fittedvalues
forecast = best_model_train.predict(start=y_z_train.shape[0], end=y_z_train.shape[0], exog = exog_dim)
forecast = y_z_train['arima_model'].append(forecast)
forecast.tail(3)
#Remove standardization
real_values = y_zscored.iloc[494:495,:]*std_y+mean_y
pred_values = pd.DataFrame(forecast.iloc[494:495])*std_y+mean_y
#MAE
mean_absolute_error(pred_values[0], real_values['Waiting_Time_Z'])
#MSE
mean_squared_error(pred_values[0], real_values['Waiting_Time_Z'])






