import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


### A. IMPORTING DATASETS ###
#A.1. Import additional features (weather data)
weather = pd.read_csv("~/Desktop/Modelos Tese/LPPT.txt")
weather_time = weather.iloc[30840:63349,]

#Pre-Processing weather data
new_weather = pd.concat((weather_time['valid'], weather_time['tmpf']),axis = 1)
new_weather = pd.concat((new_weather, weather_time['relh']),axis = 1)
new_weather = pd.concat((new_weather, weather_time['sknt']),axis = 1)
new_weather = pd.concat((new_weather, weather_time['drct']),axis = 1)
new_weather = pd.concat((new_weather, weather_time['dwpf']),axis = 1)
new_weather['tmpf_celsius'] = (new_weather['tmpf'] -32)/1.8
new_weather['dwpf_celsius'] = (new_weather['dwpf'] -32)/1.8
new_weather = new_weather.drop('tmpf', axis = 1)
new_weather = new_weather.drop('dwpf', axis = 1)
new_weather = new_weather.fillna(new_weather.mean())
new_weather['valid_new'] = new_weather['valid'].str[0:10]
new_weather=new_weather.rename(columns = {'valid_new': 'ds'})
new_weather = new_weather.groupby(by = 'ds').mean()
new_weather = new_weather.reset_index()

#A.2. Import prediction label (waiting times)
sns = pd.read_csv("~/Desktop/Modelos Tese/sns_df.csv")

#Pre-Processing waiting times
sns = sns.drop('Unnamed: 0', axis = 1)

#Choose hospital (Santa Maria) and emergency level
sns_hospital = sns[sns.Hospital.eq(218)]
sns_hospital = sns_hospital[sns_hospital.Emergency_Stage.eq(4)] #Urgency



### B. OVERALL PRE-PROCESSING ###
#B.1. Group the data by day
sns_hospital['Day'] = sns_hospital['Acquisition_Time'].str[0:10]
sns_hospital_grouped = sns_hospital.groupby(by=['Day']).mean()
day = pd.DataFrame(sns_hospital['Day'])
date = day.drop_duplicates()

#Concat days with waiting times
people = sns_hospital_grouped.iloc[:, 2:3]
date = date.reset_index()
date = date.iloc[:,1:2]
people = people.reset_index()
people = people.iloc[:,1:2]
df_prophet = pd.concat((date, people), axis = 1)


#B.2. Additional features
#Holidays
feriados = pd.DataFrame({
  'holiday': 'feriados',
  'ds': pd.to_datetime(['2017-12-01', '2017-12-08','2017-12-25',
                       '2018-01-01', '2018-03-30','2018-04-01',
                       '2018-04-25', '2018-05-01','2018-05-31',
                       '2018-06-10','2018-08-15','2018-10-05',
                       '2018-11-01','2018-04-25','2018-12-01',
                       '2018-12-08','2018-12-25','2019-01-01',
                       '2019-04-19','2019-04-21','2018-04-25','2019-04-25']),
})



### C. INITIAL PROPHET IMPLEMENTATION ###

#C.1. Prepare the data
#Rename columns appropriately 
df_prophet.columns = ['ds', 'y']
#Merge original dataframe with exogenous features dataframe
df_prophet = df_prophet.merge(new_weather, left_on='ds', right_on='ds', how = 'inner')
df_prophet['ds']=  pd.to_datetime(df_prophet['ds'])
#Fill Nans with mean value
df_prophet = df_prophet.fillna(df_prophet.mean())


#C.2.Define the model
model = Prophet(yearly_seasonality = True,
                daily_seasonality = True,
                weekly_seasonality = True,
                growth = 'linear',
                seasonality_mode = "multiplicative",
                holidays = feriados
               )
model.add_regressor('tmpf_celsius')
model.add_regressor('sknt')
model.add_regressor('dwpf_celsius')
model.add_regressor('drct')
model.add_regressor('relh')

#C.3. Fit the model
model.fit(df_prophet)

#C.4. Make predictions
#Get prediction dataframe with exogenous features 
future_added = model.make_future_dataframe(periods = 1, freq = 'D')
tmpf_celsius = df_prophet['tmpf_celsius']
sknt = df_prophet['sknt']
drct = df_prophet['drct']
dwpf_celsius = df_prophet['dwpf_celsius']
relh = df_prophet['relh']
future_added = pd.concat([future_added, tmpf_celsius,sknt,drct,dwpf_celsius,relh], axis=1)
future_added = future_added.iloc[:503,:]

#Predict
fcst_added = model.predict(future_added)

#Plots
model.plot_components(fcst_added)
plt.show()
model.plot(fcst_added)
plt.show()


### D. EVALUATE THE MODEL BASED ON 5-FOLD CROSS-VALIDATION ###
#Standardizing
mean_y = df_prophet['y'].mean()
std_y = df_prophet['y'].std()
df_prophet['y'] = (df_prophet['y']-mean_y)/std_y
#Filling nans
df_prophet = df_prophet.fillna(df_prophet.mean())

#D.1. First Fold
#Define the model
df_prophet_new = df_prophet.iloc[:490,:]
model = Prophet(yearly_seasonality = True,
                daily_seasonality = True,
                weekly_seasonality = True,
                growth = 'linear',
                seasonality_mode = "multiplicative",
                holidays = feriados
               )
model.add_regressor('tmpf_celsius')
model.add_regressor('sknt')
model.add_regressor('dwpf_celsius')
model.add_regressor('drct')
model.add_regressor('relh')
#Fit the model
model.fit(df_prophet)
#Make predictions
future_added = model.make_future_dataframe(periods = 0, freq = 'D')
tmpf_celsius = df_prophet['tmpf_celsius']
sknt = df_prophet['sknt']
drct = df_prophet['drct']
dwpf_celsius = df_prophet['dwpf_celsius']
relh = df_prophet['relh']
future_added = pd.concat([future_added, tmpf_celsius,sknt,drct,dwpf_celsius,relh], axis=1)
fcst_added = model.predict(future_added)
#Remove standardization
y_true = df_prophet['y'].values*std_y+mean_y
y_pred = fcst_added['yhat'].values*std_y+mean_y
y_pred = pd.DataFrame(y_pred)
#MAE and MSE
print('MAE: ', mean_absolute_error(y_pred.iloc[490], sns_hospital_grouped.iloc[490:491,2:3]))
mae1 = mean_absolute_error(y_pred.iloc[490], sns_hospital_grouped.iloc[490:491,2:3])
print('MSE: ',mean_squared_error(y_pred.iloc[490], sns_hospital_grouped.iloc[490:491,2:3]))
mse1 = mean_squared_error(y_pred.iloc[490], sns_hospital_grouped.iloc[490:491,2:3])

#D.2. Second Fold
#Define the Model
df_prophet_new = df_prophet.iloc[:491,:]
model = Prophet(yearly_seasonality = True,
                daily_seasonality = True,
                weekly_seasonality = True,
                growth = 'linear',
                seasonality_mode = "multiplicative",
                holidays = feriados
               )
model.add_regressor('tmpf_celsius')
model.add_regressor('sknt')
model.add_regressor('dwpf_celsius')
model.add_regressor('drct')
model.add_regressor('relh')
#Fit the model
model.fit(df_prophet)
#Make predictions
future_added = model.make_future_dataframe(periods = 0, freq = 'D')
tmpf_celsius = df_prophet['tmpf_celsius']
sknt = df_prophet['sknt']
drct = df_prophet['drct']
dwpf_celsius = df_prophet['dwpf_celsius']
relh = df_prophet['relh']
future_added = pd.concat([future_added, tmpf_celsius,sknt,drct,dwpf_celsius,relh], axis=1)
fcst_added = model.predict(future_added)
#Remove standardization
y_true = df_prophet['y'].values*std_y+mean_y
y_pred = fcst_added['yhat'].values*std_y+mean_y
y_pred=pd.DataFrame(y_pred)
#MAE and MSE
print('MAE: ', mean_absolute_error(y_pred.iloc[491], sns_hospital_grouped.iloc[491:492,2:3]))
mae2 = mean_absolute_error(y_pred.iloc[491], sns_hospital_grouped.iloc[491:492,2:3])
print('MSE: ',mean_squared_error(y_pred.iloc[491], sns_hospital_grouped.iloc[491:492,2:3]))
mse2 = mean_squared_error(y_pred.iloc[491], sns_hospital_grouped.iloc[491:492,2:3])


#D.3. Third Fold
#Define the model 
df_prophet_new = df_prophet.iloc[:492,:]
model = Prophet(yearly_seasonality = True,
                daily_seasonality = True,
                weekly_seasonality = True,
                growth = 'linear',
                seasonality_mode = "multiplicative",
                holidays = feriados
               )
model.add_regressor('tmpf_celsius')
model.add_regressor('sknt')
model.add_regressor('dwpf_celsius')
model.add_regressor('drct')
model.add_regressor('relh')
#Fit the model
model.fit(df_prophet)
#Make predictions
future_added = model.make_future_dataframe(periods = 0, freq = 'D')
tmpf_celsius = df_prophet['tmpf_celsius']
sknt = df_prophet['sknt']
drct = df_prophet['drct']
dwpf_celsius = df_prophet['dwpf_celsius']
relh = df_prophet['relh']
future_added = pd.concat([future_added, tmpf_celsius,sknt,drct,dwpf_celsius,relh], axis=1)
fcst_added = model.predict(future_added)
#Remove standardization
y_true = df_prophet['y'].values*std_y+mean_y
y_pred = fcst_added['yhat'].values*std_y+mean_y
y_pred=pd.DataFrame(y_pred)
#MAE and MSE
print('MAE: ', mean_absolute_error(y_pred.iloc[492], sns_hospital_grouped.iloc[492:493,2:3]))
mae3 = mean_absolute_error(y_pred.iloc[492], sns_hospital_grouped.iloc[492:493,2:3])
print('MSE: ',mean_squared_error(y_pred.iloc[492], sns_hospital_grouped.iloc[492:493,2:3]))
mse3 = mean_squared_error(y_pred.iloc[492], sns_hospital_grouped.iloc[492:493,2:3])

#D.4. Fourth Fold
#Define the model
df_prophet_new = df_prophet.iloc[:493,:]
model = Prophet(yearly_seasonality = True,
                daily_seasonality = True,
                weekly_seasonality = True,
                growth = 'linear',
                seasonality_mode = "multiplicative",
                holidays = feriados
               )
model.add_regressor('tmpf_celsius')
model.add_regressor('sknt')
model.add_regressor('dwpf_celsius')
model.add_regressor('drct')
model.add_regressor('relh')
#Fit the model
model.fit(df_prophet)
future_added = model.make_future_dataframe(periods = 0, freq = 'D')
tmpf_celsius = df_prophet['tmpf_celsius']
sknt = df_prophet['sknt']
drct = df_prophet['drct']
dwpf_celsius = df_prophet['dwpf_celsius']
relh = df_prophet['relh']
future_added = pd.concat([future_added, tmpf_celsius,sknt,drct,dwpf_celsius,relh], axis=1)
fcst_added = model.predict(future_added)
#Remove standardization
y_true = df_prophet['y'].values*std_y+mean_y
y_pred = fcst_added['yhat'].values*std_y+mean_y
y_pred=pd.DataFrame(y_pred)
#MAE and MSE
print('MAE: ', mean_absolute_error(y_pred.iloc[493], sns_hospital_grouped.iloc[493:494,2:3]))
mae4 = mean_absolute_error(y_pred.iloc[493], sns_hospital_grouped.iloc[493:494,2:3])
print('MSE: ',mean_squared_error(y_pred.iloc[493], sns_hospital_grouped.iloc[493:494,2:3]))
mse4 = mean_squared_error(y_pred.iloc[493], sns_hospital_grouped.iloc[493:494,2:3])

#D.5. Fifth Fold
#Define the model
df_prophet_new = df_prophet.iloc[:494,:]
model = Prophet(yearly_seasonality = True,
                daily_seasonality = True,
                weekly_seasonality = True,
                growth = 'linear',
                seasonality_mode = "multiplicative",
                holidays = feriados
               )
model.add_regressor('tmpf_celsius')
model.add_regressor('sknt')
model.add_regressor('dwpf_celsius')
model.add_regressor('drct')
model.add_regressor('relh')
#Fit the model
model.fit(df_prophet)
#Make predictions
future_added = model.make_future_dataframe(periods = 0, freq = 'D')
tmpf_celsius = df_prophet['tmpf_celsius']
sknt = df_prophet['sknt']
drct = df_prophet['drct']
dwpf_celsius = df_prophet['dwpf_celsius']
relh = df_prophet['relh']
future_added = pd.concat([future_added, tmpf_celsius,sknt,drct,dwpf_celsius,relh], axis=1)
fcst_added = model.predict(future_added)
#Remove standardization
y_true = df_prophet['y'].values*std_y+mean_y
y_pred = fcst_added['yhat'].values*std_y+mean_y
y_pred=pd.DataFrame(y_pred)
#MAE and MSE
print('MAE: ', mean_absolute_error(y_pred.iloc[494], sns_hospital_grouped.iloc[494:495,2:3]))
mae5 = mean_absolute_error(y_pred.iloc[494], sns_hospital_grouped.iloc[494:495,2:3])
print('MSE: ',mean_squared_error(y_pred.iloc[494], sns_hospital_grouped.iloc[494:495,2:3]))
mse5 = mean_squared_error(y_pred.iloc[494], sns_hospital_grouped.iloc[494:495,2:3])

total_mae = [mae1,mae2,mae3,mae4,mae5]
total_mse = [mse1,mse2,mse3,mse4,mse5]


print()
print('########################')
print()

for i in range(len(total_mae)):
    print('MAE: ',total_mae[i],' MSE: ',total_mse[i])
