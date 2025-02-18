"""BiLSTM model: build and train"""

def regmodel_creating():
    n_steps = 24
    n_features = 1

    inputs = Input(shape=(n_steps,n_features), name='input')
    main_branch = Bidirectional(LSTM(24,activation="tanh",return_sequences=True, name='lstm_1'), name='bidirectional_1')(inputs)
    main_branch = Dropout(0.15, name='dropout_1')(main_branch)
    main_branch = Bidirectional(LSTM(24,activation="tanh",return_sequences=False, name='lstm_2'), name='bidirectional_2')(main_branch)
    main_branch = Dropout(0.15, name='dropout_2')(main_branch)
    main_branch = Dense(24, activation='relu', name='dense')(main_branch)
    outputs = Dense(1, name='output')(main_branch)

    model = Model(inputs = inputs, outputs = outputs)
    # print(model.summary())
    return model

def regmodel_compile(model):
    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss = 'mse')
    return model

def regmodel_training(model, X_train, y_train, df_history, epochs=1, batch_size=100):

    history = model.fit({'input': X_train}, {'output': y_train}, epochs=epochs, batch_size=batch_size, validation_split=0)
    # history = model.fit({'input': X_train}, {'output': y_train}, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save_weights(modelbest_dir) # 都存，model.save(modelbest_dir)

    df_history_temp = pd.DataFrame(history.history)
    df_history = pd.concat([df_history, df_history_temp], axis=0)
    return df_history

def regmodel_losscurve(df_history):
    epochs = range(1, df_history.shape[0] + 1)
    loss = df_history['loss']
    val_loss = df_history['val_loss']
    plt.plot(epochs, loss, color="blue", label='loss')
    plt.plot(epochs, val_loss, color="red", label='val_loss')
    plt.title('loss curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def shuffle_list(lst):
    random.shuffle(lst)
    return lst

"""BiLSTM model: predict and evaluate"""

def regmodel_predicting(model, X_validtest, y_validtest, Datetime_validtest):
    y_pred = model.predict(X_validtest) # arr
    y_validtest = y_validtest.reshape(-1, 1) # y_validtest已經是(17520, 1)就不用reshape

    # scale回來
    traindata, validdata, testdata = splityr(df_original) # 要多加這行!!!!!!
    traindata_temp = traindata.copy()
    traindata_temp = traindata_temp.set_index("Datetime")
    traindata_temp = pd.DataFrame(traindata_temp[b])
    traindata_temp = traindata_temp.reset_index("Datetime") # 這裡traindata_temp是一棟的，上面訓練時前置作業是用一整排的
    traindata_temp, scaler_temp = normalize_train(traindata_temp, 0, 1) # 得到scaler_temp

    y_pred = y_pred.reshape(-1, 1)
    y_pred = scaler_temp.inverse_transform(y_pred)
    y_pred = y_pred.reshape(-1, 1)
    y_validtest = y_validtest.reshape(-1, 1)
    y_validtest = scaler_temp.inverse_transform(y_validtest)
    y_validtest = y_validtest.reshape(-1, 1)

    # 把時間、y_validtest、y_pred變成dataframe
    y_validtest = pd.DataFrame(y_validtest)
    y_validtest = y_validtest.rename(columns={0:'EUI_meas'})
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred.rename(columns={0:'EUI_pred'})
    df_Datetime = pd.DataFrame(Datetime_validtest)
    df_Datetime = df_Datetime.rename(columns={0:'Datetime'})

    # 把時間、y_validtest、y_pred合併回去
    df_result = pd.DataFrame()
    df_result = pd.concat([df_result,df_Datetime],axis=1)
    df_result = pd.concat([df_result,y_validtest],axis=1)
    df_result = pd.concat([df_result,y_pred],axis=1)
    df_result = df_result.set_index('Datetime')

    return df_result

def evaluation(df_data, indexname):
    y_valid = df_data['EUI_meas']
    y_pred = df_data['EUI_pred']

    # https://zhuanlan.zhihu.com/p/37663120
    errors = abs(y_pred - y_valid)
    MAPE = 100 * np.mean((errors / y_valid)) # inf因為有些y_valid=0，所以不要看這一項好了?
    NMBE = 100 * (sum(y_valid - y_pred) / (len(y_valid) * np.mean(y_valid)))
    CVRMSE = 100 * ((sum((y_valid - y_pred)**2) / (len(y_valid)-1))**(0.5)) / np.mean(y_valid)
    R2 = r2_score(y_valid, y_pred)
    MSE = np.mean((y_valid-y_pred)**2)
    RMSE = np.sqrt(mean_squared_error(y_valid, y_pred)) # RMSE = mean_squared_error(y_valid, y_pred, squared =False)
    MAE = mean_absolute_error(y_valid, y_pred)

    # iPlot
    # df_data[['EUI_meas', 'EUI_pred']].iplot()

    li_error = [MAPE, NMBE, CVRMSE, R2, MSE, RMSE, MAE]
    df_error = pd.DataFrame(li_error)
    df_error = df_error.rename(columns={0:indexname})
    df_error = df_error.rename(index={0:"MAPE", 1:"NMBE", 2:"CVRMSE", 3:"R2", 4:"MSE", 5:"RMSE", 6:"MAE"})

    return df_error
