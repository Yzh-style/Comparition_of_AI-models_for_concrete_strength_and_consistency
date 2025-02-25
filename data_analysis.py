#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[2]:


###############################################################################################
columns_to_binary = ['q_stone_1','q_stone_2','q_stone_3','q_crushed_sand','q_non_crushed_sand']
###############################################################################################
###############################################################################################
columns_to_sum_stone = ['q_stone_1', 'q_stone_2', 'q_stone_3']
columns_to_sum_sand = ['q_crushed_sand', 'q_non_crushed_sand']
###############################################################################################

file = '.\concrete_mix_design_simplified.csv'
df = pd.read_csv(file)
df = df[df['age'] >= 28]
columns_to_drop = ['probe_index', 'name', 'weight_fresh', 'weight_at_test', 'size_x', 'size_y', 'size_z', 'age','slump.1']
df = df.drop(columns=columns_to_drop)
###############################################################################################    
target_column= 'compressive_strength'
target_column1 = 'slump'


# In[3]:


def map_to_binary(df, columns_to_binary):
    for column_name in columns_to_binary:
        df[column_name] = df[column_name].map(lambda x: 1 if x != 0 else 0)
    return df


# In[4]:


def sum_columns(df, new_column, columns_to_sum):
    df[new_column] = df[columns_to_sum].sum(axis=1)
    return df


# In[5]:


sum_columns(df, 'q_stone_sum', columns_to_sum_stone)
sum_columns(df, 'q_sand_sum', columns_to_sum_sand)


# In[6]:


map_to_binary(df, columns_to_binary)
df


# In[7]:


def Encodding_cement_stone_type_fillna(df):
    if 'cement_label' in df.columns:
        df['cement_label'] = df['cement_label'].astype(str)
        df['CEM_id'] = df['cement_label'].apply(lambda x: 'CEM_I' if 'CEM_I_' in x else ('CEM_II' if 'CEM_II_' in x else ('CEM_III' if 'CEM_III_' in x else 'CEM_IV')))
        One_Hot_encoded_df = pd.get_dummies(df['CEM_id'], prefix='CEM_id')
        df['CEM_S'] = df['cement_label'].apply(lambda x: '42.5' if '42.5' in x else ('52.5' if '52.5' in x else ''))        
        df = pd.concat([df, One_Hot_encoded_df,], axis=1)
#############################################################################################################################
        roman_numeral_map = { 'I': 1, 'II':2, 'III': 3, 'IV': 4, 'N':1, 'R':2, 'N-SR':3, 'N/A-L':4, 'N/A-LL':5, 'N/B-M':6, 
                     'N/B-V':7, 'R/A-L':8, 'R/A-LL':9, 'R/A-M':10, 'R/B-LL':11, 'R/B-M':12, 'N/A-LH':13, 'N/A-S':14, 
                     'N-SR/A-S':15, 'N-SR/B-V':16
                    }
        def format_date(type_string_list):
            num = np.zeros((len(type_string_list), 1))
            i=0
            for type_string in type_string_list:
             parts = type_string.split('_')
             feature1 = 100*roman_numeral_map.get(parts[1], None)
             feature2 = float(parts[2])
             feature3 = 0.001*roman_numeral_map.get(parts[3], None)
             num[i] = feature1+feature2+feature3
             i+=1
            return num
        numeric_data = format_date(df['cement_label'])
        df['cement_label']= numeric_data
        df = df.copy() 
        df['stone_type'] = df['stone_type'].replace({'river_stone': 2, 'crushed_stone': 1,'':0})
#############################################################################################################################
        df = df.drop(columns=['CEM_id','cement_label'])
        df.fillna(0, inplace=True)
        df.replace('', 0, inplace=True) 
        return df
    else:
        print("DataFrame dosen't have 'cement_label'.")


# In[8]:


###############################################################################################
df = Encodding_cement_stone_type_fillna(df)
df
###############################################################################################


# In[9]:


def Handling_density_fresh(df, target_column):
    df_filtered = df[(df['density_fresh'] < 3000) & (df[target_column] > 1) & (df['density_fresh'] > 1500)]
    return df_filtered


# In[10]:


df=Handling_density_fresh(df,target_column)
df


# In[11]:


def useless_feature_from_Random_Forest(df,target_column,target_column1):
    df = df.drop(target_column1,axis=1)
    X = df.drop(target_column,axis=1)
    y = df[target_column] 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    param_grid = {'n_estimators': [50, 100, 20],'max_depth': [None, 2, 5],'max_features': ['auto', 'sqrt', 'log2'],
                  'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4]}    
    RFre = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=RFre, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print('best_params are:',best_params)
    final_RFre = RandomForestRegressor(**best_params)
    final_RFre.fit(X_train, y_train)
    y_pred = final_RFre.predict(X_test)
    variance = np.var(y_pred - y_test)
###############################################################################################################
    plt.figure(figsize=(15, 10))
    plt.scatter( y_test,y_pred,label='y_train and y_test')
    plt.xlabel(' y_test')
    plt.ylabel('y_pred')
    plt.text(0.5, 0.9, f'Variance: {variance:.2f}', transform=plt.gca().transAxes, fontsize=12, ha='center')
    plt.legend()
###############################################################################################################
    feature_importances = final_RFre.feature_importances_
    names = X_train.columns
    feature_importances = pd.DataFrame({'Feature': names, 'Importance': feature_importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    useless_feature = [feature_name for index, (feature_name, importance) in feature_importances.iterrows() if importance <= 0.005]
    print('useless_feature:',useless_feature)
    plt.bar(feature_importances['Feature'], feature_importances['Importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()
    return useless_feature


# In[12]:


useless_feature = useless_feature_from_Random_Forest(df,target_column,target_column1)


# In[13]:


df = df.drop(columns=useless_feature)
df


# In[14]:


def data_processing(df, target_column, target_column1):
    columns_to_sum_stone = ['q_stone_1', 'q_stone_2', 'q_stone_3']
    columns_to_sum_sand = ['q_crushed_sand', 'q_non_crushed_sand']
    columns_to_binary = ['q_stone_1', 'q_stone_2', 'q_stone_3', 'q_crushed_sand', 'q_non_crushed_sand']

    # 其他的数据处理步骤...
    sum_columns(df, 'q_stone_sum', columns_to_sum_stone)
    sum_columns(df, 'q_sand_sum', columns_to_sum_sand)
    map_to_binary(df, columns_to_binary)
    df = Encodding_cement_stone_type_fillna(df)
    df = Handling_density_fresh(df, target_column)
    useless_feature = useless_feature_from_Random_Forest(df, target_column, target_column1)
    df = df.drop(columns=useless_feature)
    return df

if __name__ == '__main__':
    file = '.\concrete_mix_design_simplified.csv'
    df = pd.read_csv(file)
    df = df[df['age'] >= 28]
    columns_to_drop = ['probe_index', 'name', 'weight_fresh', 'weight_at_test', 'size_x', 'size_y', 'size_z', 'age', 'slump.1']
    df = df.drop(columns=columns_to_drop)

    target_column = 'compressive_strength'
    target_column1 = 'slump'

    # 调用data_processing函数并传递参数
    processed_df = data_processing(df, target_column, target_column1)

    # 输出处理后的数据框架
    print(processed_df)

