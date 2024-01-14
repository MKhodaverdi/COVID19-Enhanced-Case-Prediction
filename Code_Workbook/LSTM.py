# library  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
import time
from pylab import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from bayes_opt import BayesianOptimization
from datetime import datetime, timedelta

tf.__version__
from tensorflow.compat.v1.keras.backend import get_session
tf.compat.v1.disable_v2_behavior()

# import 
covid_7 = pd.read_csv('C:/Users/khodaverdi/Desktop/New folder/data/covid/WV_COVID_data_tot.csv') 
covid_14 = pd.read_csv('C:/Users/khodaverdi/Desktop/New folder/data/covid/WV_COVID_data_tot1.csv')
popu = pd.read_csv( 'C:/Users/khodaverdi/Desktop/New folder/data/covid/IncidRtLam.csv')
svi = pd.read_csv( 'C:/Users/khodaverdi/Desktop/New folder/data/covid/2020_WestVirginia_COUNTY_SVI.csv') 
holiday = pd.read_csv('C:/Users/khodaverdi/Desktop/New folder/data/covid/Holidays.csv')
vaccine = pd.read_csv('C:/Users/khodaverdi/Desktop/New folder/data/covid/tot_vac4.csv')

# parameters
n_future = 7   
n_past = 7

# variables 
pop = popu[['county_name', 'population']]
county_names = sorted(list( covid_7.county.unique() ))

#%% data

def data_prep_new(covid_7, covid_14, popu, holiday):
    
    covid_7['date'] = pd.to_datetime(covid_7.date)
    covid_14['date'] = pd.to_datetime(covid_14.date)
    covid_7 = covid_7[ covid_7["date"] >= pd.to_datetime('20200317', format='%Y%m%d')]
    covid_7 = covid_7[ covid_7["date"] < max(covid_7['date'])- pd.to_timedelta(2,unit='d') ]
    covid_14 = covid_14[ covid_14["date"] >= max(covid_14['date'])- pd.to_timedelta(2,unit='d') ]
    covid_7 = pd.concat([covid_7, covid_14])
    
    holiday['date'] = pd.to_datetime(holiday.date)
    popu = popu.rename(columns={"county_name": "county"})
    covid = covid_7  
    covid = popu[['county', 'population']].merge(covid, on='county', how='left')
    covid = holiday.merge(covid, on='date', how='right')
    covid = covid[['date', 'county', 'incid', 'R_exp7_x', 'R_exp7_y', 'R_exp7',
		   'cum_full_180day', 'vaccine_7uptake', 'test_last7', 'recovered',
                   'population', 'weekend', 'holidays', 'hdistance', 'tholiday'
                   ]]
    covid = covid.rename(columns={ "incid": "incid_x"})
    covid = covid.reset_index(drop=True)
     
    print('data summary')
    print('Covid set shape == {}'.format(covid.shape))
    print('Featured selected: {}'.format(list(covid)[3:] ))      
    return covid



def data_scale_withincid(covid):
    cols_vars = covid.iloc[:, 2:]
    sc = StandardScaler()   
    cols_vars_scaled = sc.fit_transform(cols_vars)
    cols_vars_scaled = pd.concat([covid["date"].reset_index(drop=True), covid["county"].reset_index(drop=True), 
                                  covid["incid_x"].reset_index(drop=True), pd.DataFrame(cols_vars_scaled).reset_index(drop=True) 
                                  ], axis=1)
    return cols_vars_scaled
          


def input_output_set(covid, cols_vars_scaled, n_past, n_future):
    X_set = []
    y_set = []
    county_names = sorted(list( covid.county.unique() ))
    date_list = sorted(list( set(pd.to_datetime(covid.date)) ))
    
    for c in range(0, len(county_names) ) :
        df = cols_vars_scaled.loc[ covid["county"] == county_names[c]].sort_values(by ="date")
        for i in range(n_past, len(date_list)-n_future+1 ):  
            X_set.append(df.iloc[i - n_past:i, 3:].to_numpy())
            y_set.append(df.iloc[i + n_future-1:i + n_future, 2].to_numpy())
    X_set, y_set = np.array(X_set), np.array(y_set)
    
    print('train set summary')
    print('X_set shape == {}.'.format(X_set.shape))
    print('y_set shape == {}.'.format(y_set.shape))    
    return (X_set, y_set)



def input_pred_set(covid, cols_vars_scaled, n_past, n_future):
    X_pred = []
    county_names = sorted(list( covid.county.unique() ))
    date_list = sorted(list( set(pd.to_datetime(covid.date)) ))
    
    for c in range(0, len(county_names) ) :
        df = cols_vars_scaled.loc[ covid["county"] == county_names[c]].sort_values(by ="date")
        for i in range(  len(date_list)-n_future+1, len(date_list)+1 ):    
            X_pred.append(df.iloc[i-n_past:i, 3:].to_numpy())
    X_pred = np.array(X_pred)
    
    print('X_set shape == {}.'.format(X_pred.shape))
    return (X_pred)



def train_test_set_f_dropaweek(X_set, y_set):
    s= int(len(X_set)/len(county_names))
    for c in range(0, len(county_names) ) :
       if c == 0 :
            npx= X_set[int(s*c):int(s*(c+1)-7)]
            npy= y_set[int(s*c):int(s*(c+1)-7)]
       else:
            npx= np.concatenate((npx, X_set[int(s*c):int(s*(c+1)-7)]))
            npy= np.concatenate((npy, y_set[int(s*c):int(s*(c+1)-7)]))
    X_train, X_test, y_train, y_test = train_test_split(npx, npy, test_size=0.30)
    
    # summary
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, npx.shape, npy.shape)
    return (X_train, X_test, y_train, y_test)



#%% model

# Initializing
def generate_model(dropout, neuronCount, neuronShrink, activFun):
    model = Sequential()
    
    model.add(LSTM(units=int(neuronCount), activation=activFun, return_sequences=True, input_shape=(n_past, X_train.shape[2])))
    model.add(Dropout(dropout))
    
    model.add(LSTM(units=int(neuronCount * neuronShrink), activation=activFun, return_sequences=True))
    model.add(Dropout(dropout))
    
    model.add(LSTM(units=int(neuronCount* neuronShrink* neuronShrink), activation=activFun, return_sequences=False))  
    model.add(Dropout(dropout))
    
    model.add(Dense(units=1, activation='linear'))  #linear/ sigmoid
    model.summary()
    return model 


    
# Compiling
def fitted_model(X_test,y_test,dropout,neuronCount,neuronShrink, activFun, lr, epoch, batch):
    
    model = generate_model(dropout, neuronCount, neuronShrink, activFun)
    model.compile(optimizer = Adam(learning_rate=lr), loss='mean_absolute_error')
    es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=100, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    tb = TensorBoard('logs') 
    history = model.fit(X_train, y_train, shuffle=True, epochs=epoch, callbacks=[es, rlr, mcp, tb],
                        validation_data=(X_test, y_test), verbose=1, batch_size=batch) 
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('C:/Users/makhodaverdi/Desktop/New folder/image_loss.png')
    plt.show()
    return model, history



# evaluation
def evaluate_model(X_test,y_test,dropout,neuronCount,neuronShrink, activFun, lr, epoch, batch):
    model, history = fitted_model(dropout,neuronCount,neuronShrink, activFun, lr, epoch, batch) 
    pred = model.predict(X_test)
    mae_test = round(np.abs(np.subtract(y_test , pred)).mean(),2) 
    return -mae_test


#%% Bayesian Optimization
covid = data_prep_new(covid_7,covid_14, popu, holiday)
cols_vars_scaled = data_scale_withincid(covid)  
X_set, y_set = input_output_set(covid, cols_vars_scaled, n_past, n_future)
X_pred = input_pred_set(covid, cols_vars_scaled, n_past, n_future)
X_train, X_test, y_train, y_test = train_test_set_f_dropaweek(X_set, y_set )

def evaluate_models(dropout,neuronCount,neuronShrink, activFun, lr, epoch, batch):
    model, history = fitted_model(dropout,neuronCount,neuronShrink, activFun, lr, epoch, batch) 
    pred = model.predict(X_test)
    mae_test = round(np.abs(np.subtract(y_test , pred)).mean(),2) 
    return -mae_test

pbound = { 'dropout': (0.05, 0.045),  
           'neuronCount': (8, 64),  
           'neuronShrink': (0.01, 1),
           'lr': (0, 0.1),
           'epoch': (4, 512), 
	   'batch' : (4, 512),
	   'activFun' :("relu", "tanh")
          }
optimizer = BayesianOptimization(f=evaluate_models, pbounds=pbound, verbose=2, random_state=1)
start_time = time.time()
optimizer.maximize(init_points=10, n_iter=50,)
time_took = time.time() - start_time

dropout = optimizer.max['params']['dropout']
neuronCount = optimizer.max['params']['neuronCount']
neuronShrink = optimizer.max['params']['neuronShrink']
lr = optimizer.max['params']['lr']
epoch = optimizer.max['params']['epoch']
batch = optimizer.max['params']['batch']
activFun = optimizer.max['params']['activFun']
print(optimizer.max['params'])


#%% SHAP value
list_shap_values = []
list_test_sets = []
X_train, y_train = X_set, y_set
X_test, y_test = X_set, y_set

X_train_shap = np.random.choice(a=X_train.shape[0], size=int(X_train.shape[0]/2), replace=False)
X_test_shap = np.random.choice(a=X_test.shape[0], size=int(X_test.shape[0]), replace=False)

explainer = shap.DeepExplainer(model, X_train[X_train_shap])
shap_values = explainer.shap_values( X_test[X_test_shap] )
list_shap_values.append(shap_values)
list_test_sets.append(X_test_shap)

all_variables = np.zeros((7,4))
for x in range(shap_values[0].shape[0]) :
    all_variables = np.add(all_variables,shap_values[0][x] )
    all_variables_avg = np.divide(all_variables, shap_values[0].shape[0])
    all_variables_avg2 = all_variables_avg.sum(axis=0)

overall_importance_df = pd.DataFrame(data=all_variables_avg2, columns=['values'], index=['Rt','vaccinated','incid','Rt','vaccinated']) 
overall_importance_df = overall_importance_df.T.sort_values(by="values", axis=1, ascending=False) 
overall_importance_df = overall_importance_df.iloc[0]

        
