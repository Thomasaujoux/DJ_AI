# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:03:02 2023

@author: thoma
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce


total_data = data2[["acousticness", "danceability",	"duration_ms", "energy", "instrumentalness", "liveness", "loudness", "speechiness",	"tempo", "valence", "track_pop", "key", "mode"]]
total_data[['duration_ms']] = total_data[['duration_ms']].apply(pd.to_numeric)
total_data[['track_pop']] = total_data[['track_pop']].apply(pd.to_numeric)
total_data[['key']] = total_data[['key']].apply(pd.to_numeric)
total_data[['mode']] = total_data[['mode']].apply(pd.to_numeric)


total_data.head()
describe = total_data.describe()


corr = total_data.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(15, 20))
sns.heatmap(corr, mask=mask, vmax=1, center=0, square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .4})



total_data.hist(figsize=(20, 20))
plt.show()


def quick_scatter(df):
    '''Creates a scatterplot for each column in the dataframe, against the year column.
    Input: df (dataframe)'''
    for col in df.columns: 
        plt.scatter(np.arange(91), df[col], label=col)
        plt.legend()
        plt.xlabel(col)
        plt.xlabel('year')
        plt.title(col)
        plt.show()    
        
quick_scatter(total_data)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import itertools

#LSTM modeling libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Dropout


total_data.acousticness.plot(figsize = (18,10))
plt.xlabel('Median Acousticness', fontsize=14)
plt.ylabel('Year', fontsize=14)


roll_mean = total_data.tempo.rolling(window=2, center=False).mean()
roll_std = total_data.tempo.rolling(window=2, center=False).std()


fig = plt.figure(figsize=(14,7))
plt.plot(roll_std, color='black', label = 'Rolling Std')
plt.plot(roll_mean, color='blue', label = 'Rolling Mean')
plt.legend(loc='best')
plt.title('Rolling mean and std Over Time', size=14)
plt.show(block=False)



# Import and apply seasonal_decompose()
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(total_data.acousticness, freq = 12)

# Gather the trend, seasonality, and residuals 
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot gathered statistics
plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(total_data.acousticness, label='Original', color='blue')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend', color='blue')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality', color='blue')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals', color='blue')
plt.legend(loc='best')
plt.tight_layout()


data_diff = total_data.acousticness.diff().dropna() #1-lag differencing, notice the start year
data_diff
plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(data_diff, label='Original', color='blue')

from statsmodels.graphics.tsaplots import plot_acf
fig, ax = plt.subplots(figsize=(8,3))
plot_acf(data_diff,ax=ax, lags=20)

from statsmodels.graphics.tsaplots import plot_pacf
fig, ax = plt.subplots(figsize=(8,3))
plot_pacf(data_diff,ax=ax, lags=15)


# Let's set the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 3)
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))



ans = []
for comb in pdq:
    try:
        mod = sm.tsa.statespace.SARIMAX(total_data.acousticness,
                                        order=comb,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        output = mod.fit()
        ans.append([comb, output.aic])
        print('ARIMA {} : AIC Calculated ={}'.format(comb, output.aic))
    except:
        continue

ans_df = pd.DataFrame(ans, columns=['pdq', 'aic'])
ans_df.loc[ans_df['aic'].idxmin()]



#Plug the optimal parameter values into a new SARIMAX model
ARIMA_MODEL = sm.tsa.statespace.SARIMAX(total_data.acousticness, 
                                        order=(1, 0, 1),  
                                        enforce_stationarity=False, 
                                        enforce_invertibility=False)

# Fit the model and print results
output = ARIMA_MODEL.fit()

print(output.summary().tables[1])





output.plot_diagnostics(figsize=(10, 10))
plt.show()


pred = output.get_prediction(start=30, dynamic=False)
pred_conf = pred.conf_int()

# Get the real and predicted values
acousticness_forecasted = pred.predicted_mean             #series
acousticness_truth = total_data.acousticness[30:].T.squeeze()  #slicing the list and squeezing to a series also

# Compute the root mean square error
rmse = np.sqrt(((acousticness_forecasted - acousticness_truth) ** 2).mean())
print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 5)))



log_acousticness = np.log(total_data.acousticness)


from statsmodels.graphics.tsaplots import plot_acf
fig, ax = plt.subplots(figsize=(8,3))
plot_acf(log_acousticness,ax=ax, lags=20)

from statsmodels.graphics.tsaplots import plot_pacf
fig, ax = plt.subplots(figsize=(8,3))
plot_pacf(log_acousticness,ax=ax, lags=15)

# Let's set the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 3)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

ans = []
for comb in pdq:
    try:
        mod = sm.tsa.statespace.SARIMAX(log_acousticness,
                                        order=comb,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        output = mod.fit()
        ans.append([comb, output.aic])
        print('ARIMA {} : AIC Calculated ={}'.format(comb, output.aic))
    except:
        continue
    
    
ans_df = pd.DataFrame(ans, columns=['pdq', 'aic'])
ans_df.loc[ans_df['aic'].idxmin()]


#Plug the optimal parameter values into a new SARIMAX model
ARIMA_MODEL = sm.tsa.statespace.SARIMAX(log_acousticness, 
                                        order=(0, 1, 1),  
                                        enforce_stationarity=False, 
                                        enforce_invertibility=False)

# Fit the model and print results
output = ARIMA_MODEL.fit()

print(output.summary().tables[1])

output.plot_diagnostics(figsize=(10, 10))
plt.show()

pred = output.get_prediction(start=30, dynamic=False)
pred_conf = pred.conf_int()

# Get the real and predicted values
acousticness_forecasted = np.exp(pred.predicted_mean )            #series
acousticness_truth = total_data.acousticness[30:].T.squeeze()  #slicing the list and squeezing to a series also

# Compute the root mean square error
rmse = np.sqrt(((acousticness_forecasted - acousticness_truth) ** 2).mean())
print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 5)))




ans = []
for comb in pdq:
    try:
        mod = sm.tsa.statespace.SARIMAX(data_diff,
                                        order=comb,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        output = mod.fit()
        ans.append([comb, output.aic])
        print('ARIMA {} : AIC Calculated ={}'.format(comb, output.aic))
    except:
        continue



ans_df = pd.DataFrame(ans, columns=['pdq', 'aic'])
ans_df.loc[ans_df['aic'].idxmin()]

#Plug the optimal parameter values into a new SARIMAX model
ARIMA_MODEL = sm.tsa.statespace.SARIMAX(data_diff, 
                                        order=(0, 0, 1),  
                                        enforce_stationarity=False, 
                                        enforce_invertibility=False)

# Fit the model and print results
output = ARIMA_MODEL.fit()

print(output.summary().tables[1])

output.plot_diagnostics(figsize=(10, 10))
plt.show()



pred = output.get_prediction(start=30, dynamic=False)
pred_conf = pred.conf_int()

# Get the real and predicted values
acousticness_forecasted = pred.predicted_mean            #series
acousticness_truth = data_diff[30:]  #slicing the list and squeezing to a series also

# Compute the root mean square error
rmse = np.sqrt(((acousticness_forecasted - acousticness_truth) ** 2).mean())
print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 5)))
