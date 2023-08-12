import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 



def preprocess(project_df):
    '''
    전처리 하는 코드 
        input : dataframe - raw
        output : dataframe with preprocessing
    preprocessing
    1. 'Z_CostContact', 'Z_Revenue'제거 : 의미 없는 데이터들
    2. 'Income'에서 null값 제거 - 우리 분석의도랑 맞지 않음 & 나머지 standarad scaler
    3. 'Year_Birth' 정규화 & 표준화 
    4. 'Accept_all'만들어주기 
    5. 'Education' 각 클래스별 정수 인코딩
    6. 'Marital_Status' : 클래스별 정수 인코딩
    7. 'Num_all'만들기
    8. 'Mnt_all'만들기 
    9. 'num_every_li' 만들기
    '''
    
    ##전처리코드
    if 'Z_CostContact' in project_df.columns:
        project_df = project_df.drop(['Z_CostContact', 'Z_Revenue'], axis=1)
        project_df = project_df[project_df['Income'].notna()].reset_index()
    else : pass

    real_index = project_df['index'].values
    project_df = project_df.drop(['index'], axis=1)

    project_df['Year_Birth'] = project_df['Year_Birth'].apply(lambda x : np.abs(x-2023+1))

    project_df['Year_Birth']=StandardScaler().fit_transform(project_df['Year_Birth'].values.reshape(-1,1))

    accept_li = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp4', 'Response']
    project_df['Accept_all'] = project_df[accept_li].sum(axis=1)


    num_li = ['NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    Mnt_li = ['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts','MntGoldProds']


    def make_education_column(x:str):
        if x == '2n Cycle':
            x = 0
        elif x == 'Basic':
            x = 1
        elif x == 'Graduation':
            x = 2
        elif x == 'Master':
            x = 3
        else :
            x = 4
        return x

    project_df['Education'] = project_df['Education'].apply(lambda x : make_education_column(x))

    ## 'marital_status' 전처리
    dic_marital = {}
    for ind, status in enumerate(set(project_df['Marital_Status'])) :
        dic_marital[status] = ind

    project_df['Marital_Status'] = project_df['Marital_Status'].replace(dic_marital)

    project_df['Income']=StandardScaler().fit_transform(project_df['Income'].values.reshape(-1,1))

    project_df['Recency'] = StandardScaler().fit_transform(project_df['Recency'].values.reshape(-1,1))

    project_df['year'] = project_df['Dt_Customer'].str.split('-').str.get(0).astype(int)
    project_df['month'] = project_df['Dt_Customer'].str.split('-').str.get(1).astype(int)
    project_df['day'] = project_df['Dt_Customer'].str.split('-').str.get(2).astype(int)

    project_df['Mnt_all'] = project_df[Mnt_li].sum(axis=1)
    mnt_every_li = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'Mnt_all']

    scaler = StandardScaler()            
    scaler.fit(project_df[mnt_every_li])                  
    df_s = scaler.transform(project_df[mnt_every_li])     
    df_s = pd.DataFrame(df_s, columns = mnt_every_li)
    project_df[mnt_every_li] = df_s



    project_df['Num_all'] = project_df[num_li].sum(axis=1)
    num_every_li = ['NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Num_all'] 

    scaler = StandardScaler() 
    scaler.fit(project_df[num_every_li])                  
    df_s = scaler.transform(project_df[num_every_li])     
    df_s = pd.DataFrame(df_s, columns = num_every_li)
    project_df[num_every_li] = df_s

    return project_df