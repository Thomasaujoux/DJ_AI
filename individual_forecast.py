import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#LSTM modeling libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Dropout

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

############## Description ############
# Cette partie du projet a pour de prédire les caractéristiques du prochain son à partir des sons précédent
# L'algorithme est principalement basé sur le projet : https://github.com/oac0de/Spotify-Sound-Of-Future-Prediction/blob/main/2%20-%20Modeling%20.ipynb






############## Imoortation de la base de donnée ############
total_data = pd.read_csv('C:/Users/thoma/Documents/GitHub/DJ_AI/DJ_AI/ML/data/data.csv')
year_data = pd.read_csv('C:/Users/thoma/Documents/GitHub/DJ_AI/DJ_AI/ML/data/data_by_year.csv')

total_data.head()
year_data.head()




############## Statistiques descriptives de la base de données ##############
############## Preprocessing_&_EDA ##############

# Statistiques descriptives pures
total_data.describe()

# Corrélation entre les différentes features
# Cela nous permet de connaître les variables qui sont corrélées entre elles. 
# La correlation sera utile pour la modélisation pour la suite.
corr = total_data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(15, 20))
sns.heatmap(corr, mask=mask, vmax=1, center=0, square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .4})

# Permet de connaître le nombre de musiques par an.
plt.figure(figsize=(30,12))
sns.countplot(x=total_data.year, palette="rocket", color="k")
plt.xticks(rotation=45)
plt.show()

# Histogramme des features
total_data.hist(figsize=(20, 20))
plt.show()


# Visualisation de chaque features en fonction du temps.
def quick_scatter(df):
    '''Creates a scatterplot for each column in the dataframe, against the year column.
    Input: df (dataframe)'''
    for col in df.columns: 
        plt.scatter(df.year, df[col], label=col)
        plt.legend()
        plt.xlabel(col)
        plt.xlabel('year')
        plt.title(col)
        plt.show()  

# Les différents graphes nous donnent une répartition très intéressante de nos variables en fonction du temps.
quick_scatter(year_data)


# 
plt.figure(figsize=(30,12))
plt.xticks(rotation=45)
sns.boxplot(x="year", y="acousticness", data=total_data)
plt.xlabel("Year", size=18)
plt.ylabel("Acousticness", size=18)

# Après plusieurs manipulations qui dépendent des données 
cleaned_all_tracks = pd.read_csv('C:/Users/thoma/Documents/GitHub/DJ_AI/DJ_AI/ML/data/cleaned_all_tracks.csv')





############## Modeling ##############
yearly_df = pd.read_csv('C:/Users/thoma/Documents/GitHub/DJ_AI/DJ_AI/ML/data/yearly_merged.csv')


def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def visualize_training_results(results):
    history = results.history
    plt.figure()
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['validation loss', 'train loss'])
    plt.title('Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('loss (MSE)')
    plt.show()

def run_model(df):
    
    '''Runs an LSTM model with 5 steps, linear activation, and one dense layer
    on a date-time-indexed dataframe. 
    
    Takes a dataframe, differences it by 1-lag and splits into 90/10 train/test
    split. Then splits training and test sets into sequences required by the 
    model of 5 input steps and 1 output step. 
    
    Model is trained using 20% validation taken from the training set, and 
    tested on unseen test set. (Differenced) Predictions are then forecasted 
    5 steps into the future and "undifferenced" using np.cumsum of the original
    dataset's last value.
    
    Inputs: dataframe with a date-time index.
    
    Outputs: RMSE, loss function curve, and updated forecast dataframe with
             predicted values.
    '''
    
    #difference the column's data
    differenced_df = df.diff().dropna()
    
    #split df for train/test
    train=differenced_df[:90] 
    test=differenced_df[90:]
    
    # define input sequence
    train_seq = list(train)
    test_seq = list(test)
    n_steps = 5
    
    # split into samples
    X_train, y_train = split_sequence(train_seq, n_steps)
    X_test, y_test = split_sequence(test_seq, n_steps)
    n_features = 1
    
    #reshape from [samples, timesteps] into [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    
    #instatiate model and add layers
    model = Sequential()
    model.add(LSTM(50, activation='linear', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    #fit model with 20% validation split *of the training data*
    result = model.fit(X_train, y_train, batch_size=100, epochs=200, 
                       validation_split=(0.2), verbose=0)
    
    #predict y values with test data
    yhat = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    print('RMSE:', rmse)
    visualize_training_results(result)
    print('\n')
    
    #forecast future values
    x_input = differenced_df.tail().values
    temp_input=list(x_input)
    lst_output=[]
    i=0
    
    while(i<5):  #5 years
        if(len(temp_input)>5):  #5 steps
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            lst_output.append(yhat[0][0])
            i=i+1

        else:
            x_input = x_input.reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
            i=i+1
            
    # invert differencing of forecasts
    lst_output.insert(0, df.iloc[-1])    
    rescaled_forecast = np.cumsum(lst_output)
    rescaled_forecast = rescaled_forecast[1:]
    return rescaled_forecast




############## Entrainement sur les séquences ##############
future_years = [2021, 2022, 2023, 2024, 2025]
future_years = pd.to_datetime(future_years, format='%Y')
future_years

forecast_df = pd.DataFrame(index=future_years)
forecast_df.index.name = 'future_years'
forecast_df

rescaled_forecast = run_model(yearly_df.danceability)
rescaled_forecast
forecast_df['danceability'] = rescaled_forecast
forecast_df
