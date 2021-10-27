import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import copy

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
sns_hospital = sns_hospital[sns_hospital.Emergency_Stage.eq(1)]



### B. OVERALL PRE-PROCESSING ###

#B.1. Group data by day
sns_hospital['Day'] = sns_hospital['Acquisition_Time'].str[0:10]
sns_hospital_grouped = sns_hospital.groupby(by=['Day']).mean()

#B.2. Prepare the data
day = pd.DataFrame(sns_hospital['Day'])
date = day.drop_duplicates()
people = sns_hospital_grouped.iloc[:, 2:3]
date = date.reset_index()
date = date.iloc[:,1:2]
people = people.reset_index()
people = people.iloc[:,1:2]
df_lstm = pd.concat((date, people), axis = 1)

#B.3. Additional features
#Holidays
days = df_lstm.iloc[:,0:1]
feriados = []
for i in range(0,len(days)):
    if days.iloc[i,:].any()== '2018-01-01' or days.iloc[i,:].any()=='2017-12-01'    or days.iloc[i,:].any()=='2018-01-31'or days.iloc[i,:].any()=='2019-01-02'    or days.iloc[i,:].any()=='2017-01-31'or days.iloc[i,:].any()=='2018-01-02'    or days.iloc[i,:].any()=='2018-12-24'or days.iloc[i,:].any()=='2018-12-26'    or days.iloc[i,:].any()=='2017-12-24'or days.iloc[i,:].any()=='2017-12-26'    or days.iloc[i,:].any()=='2017-12-08'or days.iloc[i,:].any()=='2017-12-25'    or days.iloc[i,:].any()=='2018-03-30'or days.iloc[i,:].any()=='2018-04-01'    or days.iloc[i,:].any()=='2018-04-25'or days.iloc[i,:].any()=='2018-05-01'    or days.iloc[i,:].any()=='2018-05-31'or days.iloc[i,:].any()=='2018-06-10'    or days.iloc[i,:].any()=='2018-08-15'or days.iloc[i,:].any()=='2018-10-05'    or days.iloc[i,:].any()=='2018-11-01'or days.iloc[i,:].any()=='2018-04-25'    or days.iloc[i,:].any()=='2018-12-01'or days.iloc[i,:].any()=='2018-12-08'    or days.iloc[i,:].any()=='2018-12-25'or days.iloc[i,:].any()=='2019-01-01'    or days.iloc[i,:].any()=='2019-04-19'or days.iloc[i,:].any()=='2019-04-21'    or days.iloc[i,:].any()=='2018-04-25'or days.iloc[i,:].any()=='2019-04-25'    or days.iloc[i,:].any()=='2018-10-25'or days.iloc[i,:].any()=='2019-03-02'    or days.iloc[i,:].any()=='2018-01-03'or days.iloc[i,:].any()=='2019-05-05'    or days.iloc[i,:].any()=='2019-01-02'or days.iloc[i,:].any()=='2017-12-26':
        feriados.append(1)
    else:
        feriados.append(0)
holidays = pd.concat((pd.DataFrame(days),pd.DataFrame(feriados)),axis = 1)

#B.4. Merge original dataframe with exogenous features dataframe
cov_final = new_weather.merge(df_lstm, left_on='Day', right_on='Day')

#B.5. Create dataframes for additional features
#Temperature
tmp = cov_final.drop('Waiting_Time', axis =1 )
tmp = tmp.drop('relh', axis =1 )
tmp = tmp.fillna(tmp.mean())
#Humidity
relh = cov_final.drop('Waiting_Time', axis =1 )
relh = relh.drop('tmpf_celsius', axis =1 )
relh = relh.fillna(0)


#B.6. Split into Train and Test Sets
test_data_size=13 #(8+5)
#Original data
train_data = df_lstm[:-test_data_size]
test_data = df_lstm[-test_data_size:]
#Temperature
train_data_tmp = tmp[:-test_data_size]
test_data_tmp = tmp[-test_data_size:]
#Humidity
train_data_relh = relh[:-test_data_size]
test_data_relh = relh[-test_data_size:]
#Holidays
train_data_fer = holidays[:-test_data_size]
test_data_fer = holidays[-test_data_size:]

#Convert to list
test_data = list(test_data.iloc[:,1])
test_data_tmp = list(test_data_tmp.iloc[:,1])
test_data_relh = list(test_data_relh.iloc[:,1])
test_data_fer = list(test_data_fer.iloc[:,1])

#B.7. Extra: standardize the data
mean_temp = train_data_tmp['tmpf_celsius'].mean()
std_temp = train_data_tmp['tmpf_celsius'].std()
mean_relh = train_data_relh['relh'].mean()
std_relh = train_data_relh['relh'].std()
relh['relh'] = (relh['relh']-relh['relh'].mean())/relh['relh'].std()
tmp['tmpf_celsius'] = (tmp['tmpf_celsius']-tmp['tmpf_celsius'].mean())/tmp['tmpf_celsius'].std()

mean_wait = train_data['Waiting_Time'].mean()
std_wait = train_data['Waiting_Time'].std()
df_lstm['Waiting_Time'] = (df_lstm['Waiting_Time']-df_lstm['Waiting_Time'].mean())/df_lstm['Waiting_Time'].std()


### C. LSTM IMPLEMENTATION ###

#C.1. Prepare the data
#Original data
train_data_lstm = []
for i in range(len(train_data)):
    train_data_lstm.append([train_data.iloc[i][1]])
#Temperature
train_data_lstm_tmp = []
for i in range(len(train_data_tmp)):
    train_data_lstm_tmp.append([train_data_tmp.iloc[i][1]])
#Humidity
train_data_lstm_relh = []
for i in range(len(train_data_relh)):
    train_data_lstm_relh.append([train_data_relh.iloc[i][1]])
#Holidays
train_data_lstm_fer = []
for i in range(len(train_data_fer)):
    train_data_lstm_fer.append([train_data_fer.iloc[i][0]])

#C.3. Convert to a tensor
data_pytorch = torch.FloatTensor(train_data_lstm).view(-1)
data_pytorch_tmp = torch.FloatTensor(train_data_lstm_tmp).view(-1)
data_pytorch_relh = torch.FloatTensor(train_data_lstm_relh).view(-1)
data_pytorch_fer = torch.FloatTensor(train_data_lstm_fer).view(-1)

#C.4. Create input sequence
def create_input_sequences(input_data,tmp,relh,fer, tw):
    input_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_seq_tmp = tmp[i:i+tw]
        train_seq_relh = relh[i:i+tw]
        train_seq_fer = fer[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        input_seq.append((train_seq ,train_seq_tmp,train_seq_relh,train_seq_fer,train_label))
    return input_seq

train_input_seq = create_input_sequences(data_pytorch, data_pytorch_tmp, data_pytorch_relh,data_pytorch_fer,7)


#C.5. LSTM Class

class LSTM(nn.Module):
    def __init__(self, input_size = 4, hidden_layer_size= 50, output_size = 1): #Tune the hidden layer size manually
        super().__init__() #input_size = no. of features, hidden_size = number of neurons in the layer
        
        self.hidden_layer_size = hidden_layer_size
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size), 
                           torch.zeros(1,1,self.hidden_layer_size))
        
    def forward(self,input_seq,tmp,relh,fer):
        
        x = torch.cat((input_seq, tmp,relh,fer),dim = -1)

        lstm_out, self.hidden_cell = self.lstm(x.view(len(input_seq), 1, -1), self.hidden_cell)
                
        predictions = self.linear(lstm_out)
                
        return predictions[-1]


#Fit the LSTM and experiment with different loss functions
model = LSTM()
#loss_function = nn.L1Loss()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
print(model)

#Get total number of parameters
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
pytorch_total_params


#C.6. Train the model
epochs = 100
min_loss=900
count=0
predictions = []
for i in range(epochs):
    predictions = []
    for seq,tmp, relh,fer,labels in train_input_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1,1,model.hidden_layer_size),
                            torch.zeros(1,1,model.hidden_layer_size))
        
        y_pred = model(seq,tmp,relh,fer)
        
        predictions.append(y_pred.item())
        
        labels= labels.view(-1,1)
        
        single_loss = loss_function(y_pred,labels)
        single_loss.backward()
        optimizer.step()
        
    #early stopping    
    if min_loss > single_loss.item():
        count = 0
        min_loss = single_loss.item()
        print('min_loss:',min_loss)
        sd_clone = copy.deepcopy(model.state_dict())
        
    elif min_loss <=single_loss.item():
        count += 1
        print('sem melhoria: ',count)
            
    if count>5:
        break
        
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')



#C.7. Make predictions (use the last 7 days to predict the following 5 days)
fut_pred = 13
#Original Data
test_inputs = data_pytorch[-7:].tolist()
print(test_inputs) #last 7 values on the training set
#Temperature
test_inputs_tmp = data_pytorch_tmp[-7:].tolist()
print(test_inputs_tmp)
#Humidity
test_inputs_relh = data_pytorch_relh[-7:].tolist()
print(test_inputs_relh)
#Holidays
test_inputs_fer = data_pytorch_fer[-7:].tolist()
print(test_inputs_fer)


### D. EVALUATE THE MODEL ###
model.load_state_dict(sd_clone) #saves best model 
model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-7:])
    tmp = torch.FloatTensor(test_inputs_tmp[-7:])
    relh = torch.FloatTensor(test_inputs_relh[-7:])
    fer = torch.FloatTensor(test_inputs_fer[-7:])
    
    with torch.no_grad():
        model.hidden = (torch.zeros(1,1,model.hidden_layer_size),
                       torch.zeros(1,1,model.hidden_layer_size))
        test_inputs.append(model(seq,tmp,relh,fer).item())

#Get predictions
preds = pd.DataFrame(test_inputs[-5:])
#Remove standardization
preds = preds*std_wait+mean_wait
#Real observed values
real = pd.DataFrame(test_data[0:5])

#MAE
print('MAE: ',mean_absolute_error(real, preds))
#MSE
print('MSE: ',mean_squared_error(real, preds))


