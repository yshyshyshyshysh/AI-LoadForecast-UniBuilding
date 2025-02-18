"""Build clustering model"""

def kmeans_clustering(df_data, selected_K):

    # Convert data into the form to be thrown into kmeans
    df_data = df_data.set_index('Datetime')
    df_data = df_data.T
    df_data = df_data.values

    # Training
    kmeans = KMeans(n_clusters=selected_K, n_init=10).fit(df_data)
    
    # Predicting
    df_dataresult = kmeans.predict(df_data) #kmeans.labels_

    # Build dictionary
    cluster_buildingdict = {}
    for i in range(selected_K): # 先建立空字典，只是紀錄幾類
        cluster_buildingdict.update({i:[]})
    for i in range(len(available_buildingli)): # 接著把建築放進分類中
        cluster_temp = df_dataresult[i]
        cluster_buildingdict[cluster_temp].append(available_buildingli[i])

    return cluster_buildingdict, kmeans

"""Decide the number of clusters firstly"""

def kmeans_choosek(df_data):
    
    # 字體
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = "STIXGeneral"

    # Convert data into the form to be used in KMeans
    df_data = df_data.set_index('Datetime')
    df_data = df_data.T
    df_data = df_data.values

    # Error Sum of Squares Plot (Elbow method)
    clusters = 10
    k_range = range(2, clusters + 1)  # 2~10
    distortions = []
    silhouette_scores = []
    for i in k_range:
        kmeans = KMeans(n_clusters=i, n_init=10, random_state=0).fit(df_data)
        distortions.append(kmeans.inertia_)
        if i > 1:  # Silhouette Score is only defined for more than one cluster
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(df_data, labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(-1)  # Not defined for k=1
    
    print(f'distortions = {distortions}')
    print(f'silhouette_scores = {silhouette_scores}')
    return k_range, distortions, silhouette_scores

"""Save dictionary"""

def save_dictpkl(dict_temp, name_temp):
    path_temp = r'output/model_results/'
    path_temp += name_temp + '.pkl'
    with open(path_temp, 'wb') as f:
        pickle.dump(dict_temp, f)
    print(path_temp)

def read_dictpkl(name_temp):
    path_temp = r'output/model_results/'
    path_temp += name_temp + '.pkl'
    with open(path_temp, 'rb') as f:
        dict_temp = pickle.load(f)
    return dict_temp
