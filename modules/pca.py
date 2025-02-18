"""PCA to reduce data to 2 dimension"""

def pca_func(df_data):
    df_data = df_data.set_index('Datetime')
    df_data = df_data.T
    pca_obj = PCA(n_components=2)  # 將變數名稱改為 pca_obj
    df_data = pca_obj.fit_transform(df_data)
    pca_obj.explained_variance_ratio_ = pca_obj.explained_variance_ratio_.round(3)

    df_pca = pd.DataFrame(data=df_data, columns=['principal component 1', 'principal component 2'])
    df_pca = pd.concat([pd.DataFrame(available_buildingli), df_pca], axis=1)
    df_pca = df_pca.rename(columns={0: 'target'})

    pca_ratio = pca_obj.explained_variance_ratio_
    print('Principal Component 1 (' + str(pca_ratio[0]) + ')')
    print('Principal Component 2 (' + str(pca_ratio[1]) + ')')

    return df_pca, pca_ratio, pca_obj
