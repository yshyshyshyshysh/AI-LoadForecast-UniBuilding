"""Data Splitting: divided into training, validation and test sets according to year"""

def splityr(df_data):

    columnnameli = list(df_data.columns)
    df_Datetime = df_data["Datetime"] #.iloc[:,0:1]

    df_data = df_data.set_index('Datetime')

    split_date_b1 = '2018-12-31 23:00:00'
    split_date_a1 = '2018-12-31 00:00:00' # 要提前一天
    split_date_b2 = '2019-12-31 23:00:00'
    split_date_a2 = '2020-01-20 00:00:00' # 要提前一天

    traindata = df_data.loc[:split_date_b1,:]
    validdata = df_data.loc[split_date_a1:split_date_b2,:]
    testdata = df_data.loc[split_date_a2:,:]

    traindata = traindata.reset_index()
    validdata = validdata.reset_index()
    testdata = testdata.reset_index()

    return traindata, validdata, testdata

"""Data Normalization"""

def normalize_train(df_data, lowerbound, upperbound):
    df_data = df_data.set_index('Datetime')
    columnname = list(df_data.columns)
    indexname = list(df_data.index)

    scaler = MinMaxScaler(feature_range=(lowerbound, upperbound)).fit(df_data)
    df_data = scaler.transform(df_data)
    df_data = pd.DataFrame(df_data, index=indexname, columns=columnname)
    # print(df_data.describe())

    df_data = df_data.reset_index()
    df_data = df_data.rename(columns={"index":"Datetime"})
    return df_data, scaler

def normalize_validtest(df_data, scaler, lowerbound, upperbound):
    df_data = df_data.set_index('Datetime')
    columnname = list(df_data.columns)
    indexname = list(df_data.index)

    df_data = scaler.transform(df_data)
    df_data = pd.DataFrame(df_data, index=indexname, columns=columnname)
    # print(df_data.describe())

    df_data = df_data.reset_index()
    df_data = df_data.rename(columns={"index":"Datetime"})
    return df_data

"""Sliding Window: reshape for sliding window"""

def reshapearr(b, traindata, validdata, testdata):
    X_train = pd.DataFrame(traindata.set_index('Datetime')[b].copy())
    X_valid = pd.DataFrame(validdata.set_index('Datetime')[b].copy())
    X_test = pd.DataFrame(testdata.set_index('Datetime')[b].copy())
    y_train = pd.DataFrame(traindata.set_index('Datetime')[b].copy())
    y_valid = pd.DataFrame(validdata.set_index('Datetime')[b].copy())
    y_test = pd.DataFrame(testdata.set_index('Datetime')[b].copy())

    # 先把時間存下來
    y_train = y_train.reset_index()
    Datetime_train = y_train.iloc[:,0:1]
    y_train = y_train.set_index('Datetime')
    y_valid = y_valid.reset_index()
    Datetime_valid = y_valid.iloc[:,0:1]
    y_valid = y_valid.set_index('Datetime')
    y_test = y_test.reset_index()
    Datetime_test = y_test.iloc[:,0:1]
    y_test = y_test.set_index('Datetime')

    # 然後把舊的改名字
    X_train_old = X_train.copy()
    X_valid_old = X_valid.copy()
    X_test_old = X_test.copy()
    y_train_old = y_train.copy()
    y_valid_old = y_valid.copy()
    y_test_old = y_test.copy()
    Datetime_train_old = Datetime_train.copy()
    Datetime_valid_old = Datetime_valid.copy()
    Datetime_test_old = Datetime_test.copy()

    # 開始做新的
    seq_len = 24
    X_train = []
    X_valid = []
    X_test = []
    y_train = []
    y_valid = []
    y_test = []
    Datetime_train = []
    Datetime_valid = []
    Datetime_test = []

    for i in range(seq_len, len(traindata)):
        X_train.append(X_train_old.iloc[i-seq_len : i])
        y_train.append(y_train_old.iloc[i, 0])
        Datetime_train.append(Datetime_train_old.iloc[i, 0])

    for i in range(seq_len, len(validdata)):
        X_valid.append(X_valid_old.iloc[i-seq_len : i])
        y_valid.append(y_valid_old.iloc[i, 0])
        Datetime_valid.append(Datetime_valid_old.iloc[i, 0])

    for i in range(seq_len, len(testdata)):
        X_test.append(X_test_old.iloc[i-seq_len : i])
        y_test.append(y_test_old.iloc[i, 0])
        Datetime_test.append(Datetime_test_old.iloc[i, 0])

    # 把list轉成array
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)
    Datetime_train = np.array(Datetime_train)
    Datetime_valid = np.array(Datetime_valid)
    Datetime_test = np.array(Datetime_test)
    print('X_train.shape = ',X_train.shape)
    print('X_valid.shape = ',X_valid.shape)
    print('X_test.shape = ', X_test.shape)
    print('y_train.shape = ',y_train.shape)
    print('y_valid.shape = ',y_valid.shape)
    print('y_test.shape = ',y_test.shape)
    print('Datetime_train.shape = ',Datetime_train.shape)
    print('Datetime_valid.shape = ',Datetime_valid.shape)
    print('Datetime_test.shape = ',Datetime_test.shape)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, Datetime_train, Datetime_valid, Datetime_test

"""Save and read reshape results (to save modelling time)"""

def save_arrnpy(arr_temp, name_temp):
    path_temp = r'output/reshapearr/'
    path_temp += name_temp + '.npy'
    np.save(path_temp, arr_temp)
    print(name_temp)

def read_arrnpy(name_temp):
    path_temp = r'output/reshapearr/'
    path_temp += name_temp + '.npy'
    arr_temp = np.load(path_temp)
    return arr_temp

"""Separate different epidemic stages"""

def splitcovid_period(df_data):
    li_pidata = []

    # 先用年份分成train、valid跟test
    # 注意: 數據本身就有設index
    split_date_b1 = '2020-03-01 23:00:00'
    split_date_a1 = '2020-03-02 00:00:00'
    split_date_b2 = '2020-05-17 23:00:00'
    split_date_a2 = '2020-05-18 00:00:00'
    split_date_b3 = '2021-05-11 23:00:00'
    split_date_a3 = '2021-05-12 00:00:00'
    split_date_b4 = '2021-07-12 23:00:00'
    split_date_a4 = '2021-07-13 00:00:00'
    split_date_b5 = '2021-10-12 23:00:00'
    split_date_a5 = '2021-10-13 00:00:00'

    p1data = df_data.loc[:split_date_b1,:]
    p2data = df_data.loc[split_date_a1:split_date_b2,:]
    p3data = df_data.loc[split_date_a2:split_date_b3,:]
    p4data = df_data.loc[split_date_a3:split_date_b4,:]
    p5data = df_data.loc[split_date_a4:split_date_b5,:]
    p6data = df_data.loc[split_date_a5:,:]

    li_pidata.append(p1data)
    li_pidata.append(p2data)
    li_pidata.append(p3data)
    li_pidata.append(p4data)
    li_pidata.append(p5data)
    li_pidata.append(p6data)

    return li_pidata

"""ACF"""

def cal_acf(x, nlags):
    x = np.array(x)
    mean_x = np.mean(x)
    length_x = x.shape[0]
    c_0 = np.mean((x-mean_x) **2)
    c_k = np.sum((x[:(length_x-nlags)] - mean_x) * (x[nlags:] - mean_x)) / length_x
    r_k = c_k / c_0
    return r_k
