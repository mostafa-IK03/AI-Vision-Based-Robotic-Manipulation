
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# loading the data sets
# df = pd.read_csv(r'C:\Users\Dell\Desktop\798k_proj\Universal_IK_Solver\dataset_generation\datasets_2DOF\2dof_dataset.csv',  encoding = 'utf8')
df = pd.read_csv(r'C:\Users\Dell\Desktop\798k_proj\An-ML-based-approach-for-solving-inverse-kinematic-of-a-6DOF-robotic-arm-main\datasets_6DOF\dataset_myplace_with_constraints_no6_merged.csv',   encoding = 'utf8')
df = df.drop(['Unnamed: 0'], axis = 1)

# number of input features
size = 12
# number of outputs
angles = 6 

cross_val = 0

"""""

Train-Test Split vs. K-Fold Cross-Validation:
    If cross_val == 0: split once with train_test_split
    If cross_val == 1: do K-Fold Cross-Validation with K=5 splits

"""""

if cross_val == 0 :
    
    x_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    y_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    
    X = df.iloc[:,:size]; 
    y = df.iloc[:,size:]; 
    
    X_s = x_scaler.fit_transform(X)  # scale inputs to [-1,1]
    y_s = y_scaler.fit_transform(y)  # scale outputs to [-1,1]
    
    X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2)
    
    X_train = pd.DataFrame(X_train); X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train); y_test = pd.DataFrame(y_test)
    
    """""

    LSTMs expect 3D inputs: [batch, timesteps, features]
    Here we are using a single timestep (timesteps=1), effectively reducing the LSTM to learn a static mapping which i still valid, though not leveraging temporal context provide by LSTM

    """""

    X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]) )
    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]) )
    
    model = Sequential()

    """""
    Four stacked LSTM layers, progressively increasing hidden units.
        return_sequences=True makes each LSTM layer output a sequence, except the last one.
        Final Dense layers map to 6 joint angles with linear activation.

    """""
    model.add(LSTM(64, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    # model.add(LSTM(256, activation='relu')) #added
    model.add(LSTM(256, activation='relu'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(angles, activation='linear'))
    # model.add(LSTM(512, activation='relu', return_sequences=True))
    # model.add(Dropout(0.3))  # Add dropout layer with 30% drop rate
    # model.add(LSTM(512, activation='relu'))  # Last LSTM without return_sequences
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.2)) 
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(angles, activation='linear'))

    
    callbacks = [EarlyStopping(monitor='val_acc', patience = 10)]
    
    optimizer = Adam(lr=0.0005)
    model.compile(
        loss='mae',
        optimizer=optimizer,
        metrics=['accuracy'])
    
    model_f=model.fit(X_train,
              y_train,
              callbacks = callbacks,
              epochs=200,
              validation_split = 0.2)
    
    y_pred = model.predict(X_test)


    
    #           LSTM end
        
    # Get training and test loss histories
    training_loss = model_f.history['loss']
    test_loss = model_f.history['val_loss']
    
    # Get training and test accuracy histories
    training_acc = model_f.history['acc']
    test_acc = model_f.history['val_acc']
    
    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    
    # Visualize loss history
    plt.figure()
    plt.title('Loss')
    plt.plot(epoch_count, training_loss)
    plt.plot(epoch_count, test_loss)
    plt.legend(['Train', 'Test'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.show()
    
    # Visualize accuracy history
    plt.figure()
    plt.title('Acuracy')
    plt.plot(epoch_count, training_acc)
    plt.plot(epoch_count, test_acc)
    plt.legend(['Train', 'Test'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy value')
    plt.show()
    
    y_pred = pd.DataFrame(y_scaler.inverse_transform(y_pred))
    y_test = pd.DataFrame(y_scaler.inverse_transform(y_test))

     # Save model and scalers 
    model.save("ik_lstm_6_nd.h5")
    print("Model saved successfully.")

    
    joblib.dump(x_scaler, 'x_scaler_6_nd.pkl')
    joblib.dump(y_scaler, 'y_scaler_6_nd.pkl')
    print("Scalers saved successfully.")
    
    result_mse = []
    result_mae = []
    
    for i in range(angles):
    
       mse = mean_squared_error(y_test.iloc[:,i], y_pred.iloc[:,i])
       rmse = sqrt(mse) 
       mae = mean_absolute_error(y_test.iloc[:,i], y_pred.iloc[:,i])
       result_mse.append(mse)
       result_mae.append(mae)
    
    print("RMSE", result_mse)
    print("MAE", result_mae)
    
    
    X_test_r = pd.DataFrame(x_scaler.inverse_transform( pd.DataFrame(np.reshape(X_test,(X_test.shape[0],X_test.shape[2]))) ))
    
    y_pred.to_csv('y_pred_6c.csv')
    y_test.to_csv('y_test_6c.csv')
    X_test_r.to_csv('X_test_6c.csv')

else:
    # Inputs and outputs are scaled between [-1, 1]  This improves training stability.

    scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    
    X = df.iloc[:,:size]; 
    y = df.iloc[:,size:]; 
    
    X_scaler = scaler.fit(X)
    X_s = scaler.transform(X)
#    X_s = pd.DataFrame(X)
    
    y_scaler = scaler.fit(y)
    y_s = scaler.transform(y)
#    y_s = pd.DataFrame(y)
    
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
        #           cross-validation
    
    
    kfold = KFold(5, True, 1)
    count = 0
    accuracies = []
        
    for train, test in kfold.split(df):
    #	print('train: %s, test: %s' % (df[train], df[test]))
        count = count +1
        print("K_FOLD ITERATION NUMBER ",count)
        X_train = X_s[train[0]:train[-1],:]
        X_test = X_s[test[0]:test[-1],:]
        y_train = y_s[train[0]:train[-1]]
        y_test = y_s[test[0]:test[-1]] 
    
        #           LSTM start
        
        X_test = pd.DataFrame(X_test)
        X_train = pd.DataFrame(X_train)
        
        X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]) )
        X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]) )
        
        model = Sequential()
        model.add(LSTM(64, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
        #model.add(Dropout(0.1))
        model.add(LSTM(128, activation='relu', return_sequences=True))
        #model.add(Dropout(0.1))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(LSTM(256, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(angles, activation='linear'))
        
        callbacks = [EarlyStopping(monitor='val_acc', patience = 10)]
        
        model.compile(
            loss='mae',  
            optimizer='Adam',
            metrics=['accuracy'])
        
        model_f=model.fit(X_train,
                  y_train,
                  callbacks = callbacks,
                  epochs=100,
                  validation_split = 0.2)
        
        y_pred = model.predict(X_test)
        
        #model = load_model()
        #model.save()
        # Save model and scalers 
        model.save("ik_lstm_n_2d.h5")
        print("Model saved successfully.")

    
        joblib.dump(X_scaler, 'x_scaler_n_2d.pkl')
        joblib.dump(y_scaler, 'y_scaler_n_2d.pkl')
        print("Scalers saved successfully.")
        
        #           LSTM end
        
        
        # Get training and test loss histories
        training_loss = model_f.history['loss']
        test_loss = model_f.history['val_loss']
        
        # Get training and test accuracy histories
        print(model_f.history.keys())
        training_acc = model_f.history['acc']
        test_acc = model_f.history['val_acc']
        
        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        
        # Visualize loss history
        plt.figure()
        plt.title('Loss')
        plt.plot(epoch_count, training_loss)
        plt.plot(epoch_count, test_loss)
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.show()
        
        # Visualize accuracy history
        plt.figure()
        plt.title('Acuracy')
        plt.plot(epoch_count, training_acc)
        plt.plot(epoch_count, test_acc)
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy value')
        plt.show()
        
        y_pred = pd.DataFrame(y_scaler.inverse_transform(y_pred))
        y_test = pd.DataFrame(y_scaler.inverse_transform(y_test))

        # model.save("ik_lstm.h5")
        # print("Model saved successfully.")

        
        
        result_mse = []
        result_mae = []
        for i in range(angles):
        
           mse = mean_squared_error(y_test.iloc[:,i], y_pred.iloc[:,i])
           rmse = sqrt(mse) 
           mae = mean_absolute_error(y_test.iloc[:,i], y_pred.iloc[:,i])
           result_mse.append(mse)
           result_mae.append(mae)
        
        #print("RMSE", result_mse)
        print("MAE", result_mae)
        
        accuracies.append(result_mae)
    
    mean_acc = []
    std_acc = []
    accuracies = np.array(accuracies)
    for i in range(len(accuracies[0])):
        mean_acc.append(np.mean(accuracies[:,i]))
        std_acc.append(np.std(accuracies[:,i]))
    print('mean', mean_acc)
    print('std', std_acc)


# import joblib

# joblib.dump(x_scaler, 'x_scaler.pkl')
# joblib.dump(y_scaler, 'y_scaler.pkl')