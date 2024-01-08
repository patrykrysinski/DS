from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import os
import plotly.express as px
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

def kaggle_download(user, dataset):

    """ 
    Function to load dataset from Kaggle. The Kaggle API needs to be installed and configured
    user - user from which data should be loaded
    dataset - name of the dataset to import
    """

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(f"{user}/{dataset}")
    zips = [i for i in os.listdir() if i.endswith('zip')]
    for zip in zips:
        zf = ZipFile(zip)
        zf.extractall() 
        zf.close()

def bubble_plot(df,x,y,ids_col,groupper,list_to_change_location,agg_func = np.mean) -> pd.DataFrame:
    """
    Function to group data by groupper and calculate aggregate function for x and y as well as number of record in each group.
    Then scatter / bubble plot is made on aggregated data 
    """
    prices = df.groupby(groupper).agg({y:agg_func,x: agg_func ,ids_col: np.size}).rename({ids_col: 'No of ads'},axis=1).reset_index()
    prices['Age'] = np.round(prices['Age'])
    print(prices)
    text_pos = ['bottom center' if i in list_to_change_location  else 'top center' for i in prices[groupper] ]
    fig = px.scatter(prices, x='Price',y='Age',size='No of ads',text=groupper)
    fig.update_traces(textposition=text_pos)
    fig.update_layout(
    title_text=f"Cars' {x} and {y} by {groupper}"
    )
    fig.show()
    return prices
    
def barplot_share(df, column):
    agg_df = df.groupby(column).count().reset_index().iloc[:,:2]
    agg_df.columns = [column,'number']
    agg_df['share'] =  np.round(agg_df['number']/ agg_df['number'].sum(),3)
    agg_df = agg_df.sort_values(by='share', ascending = False)
    return px.bar(data_frame=agg_df, x=column,y='share',title=f"Share by {column}" )

def remove_space(df):
    data_cols_old = df.columns
    data_cols_new  = [i.replace(' ','_') for i in data_cols_old]
    to_change = dict(map(lambda i,j : (i,j) , data_cols_old,data_cols_new))
    data = df.rename(to_change, axis=1)
    return data

def encoding(df,cols_to_one_hot, cols_to_avg,y='Price'):
    df = pd.get_dummies(data=df, columns=cols_to_one_hot, drop_first=True)
    for i in cols_to_avg:
        encoding = df[[y,i]].groupby(i).mean().sort_values(by=y).reset_index()
        encoding[f'{i}_encoded'] = np.arange(0,encoding.shape[0])
        del encoding[y]
        df = df.merge(encoding, on=i)
    return df


def objective(trial, train_x, train_y,obj,seed,numerical_features,cat_features,fold_no = 3):
    hyperparams = {'objective':trial.suggest_categorical('objective',['regression','regression_l1']),
                    'boosting': trial.suggest_categorical('boosting',['gbdt','dart']),
                    'learning_rate' : trial.suggest_float('learning_rate',0.05,0.8),
                    'num_iterations':trial.suggest_int('num_iterations',10,1000),
                    'max_depth': trial.suggest_int('max_depth',2,15),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',10,50),
                    'bagging_fraction': trial.suggest_float('bagging_fraction',0.4,1),
                    'feature_fraction': trial.suggest_float('bagging_fraction',0.4,1),
                    'lambda_l1': trial.suggest_float('lambda_l1',0,10),
                    'lambda_l2': trial.suggest_float('lambda_l2',0,10),
                    'num_leaves':  trial.suggest_int('num_leaves',5,100),
                    'linear_lambda': trial.suggest_float('linear_lambda',0,10),
                    'min_gain_to_split': trial.suggest_float('min_gain_to_split',0,10),
                    'saved_feature_importance_type': 1,
                    'drop_rate':trial.suggest_float('drop_rate',0.05,1),
                    'max_drop': trial.suggest_int('max_drop',20,100),
                    'seed': seed,
                    'force_col_wise': True ,
                    'make_var': trial.suggest_categorical('make_var',['make_new_1_encoded','make_new_2_encoded']) }
    if ('make_new_1_encoded' in cat_features) &  ('make_new_2_encoded' in cat_features):
        cat_features.remove(hyperparams['make_var'])
    model_feature = []
    model_feature.extend(cat_features)
    model_feature.extend(numerical_features)
    params_to_model = {key: value for key, value in hyperparams.items() if key != 'make_var'}
    train_lgb = lgb.Dataset(data=train_x[model_feature], label=train_y, categorical_feature=cat_features)
    model = lgb.cv(params=params_to_model,
                   train_set=train_lgb,
                   #num_boost_round=num_iterations,
                   seed =seed,
                   nfold=fold_no,
                   metrics=obj
    )
    
    return model[f'valid {obj}-mean'][hyperparams['num_iterations']-1]

def evaluation(pred_train, train_y, pred_test, test_y):
    """
    Function to create evaluation summary with the following metrics:
    - r2
    - mean absolute error (mae)
    - mean absolute percentage error (mape)
    - root mean squared error (rmse) 
    """
    r2_train = r2_score(train_y, pred_train)
    r2_test = r2_score(test_y, pred_test)
    mae_train = mean_absolute_error(train_y, pred_train)
    mae_test = mean_absolute_error(test_y, pred_test)
    mape_train = mean_absolute_percentage_error(train_y, pred_train)
    mape_test = mean_absolute_percentage_error(test_y, pred_test)
    rmse_train = np.sqrt(mean_squared_error(train_y, pred_train))
    rmse_test = np.sqrt(mean_squared_error(test_y, pred_test))
    results = pd.DataFrame(columns=['dataset','r2','mae','mape','rmse'])
    results.loc[0,:] = ['train',r2_train,mae_train,mape_train,rmse_train]
    results.loc[1,:] = ['test',r2_test,mae_test,mape_test,rmse_test]
    print(results)
    plt.scatter(train_y, pred_train)
    plt.title('Actual vs. predicted values - train set')
    plt.plot([min(train_y), max(train_y)],
         [min(train_y), max(train_y)],
         color='red', linestyle='--')
    plt.show()
    plt.scatter(test_y, pred_test)
    plt.title('Actual vs. predicted values - test set')
    plt.plot([min(test_y), max(test_y)],
         [min(test_y), max(test_y)],
         color='red', linestyle='--')
    plt.show()


def calc_split_summary(df,all_features, calc_type = np.mean, y='Price',x = 'Ad ID'):
    all_diff = []
    min_no_of_rec = []
    for feat in all_features:
        agg = df[[feat,y,x]].groupby(feat).agg({y:calc_type, x:np.size}).rename({x: 'no of records'}, axis=1)
        min_no_of_rec.append(agg['no of records'].min())
        diff = round(agg.loc[1,y] / agg.loc[0,y] * 100)
        all_diff.append(diff)
        print(agg)
        print(f" Difference between value 1 and 0 equals to: {diff}%")
        print('##############\n')

def info_splitter(df, column_to_split,split = ','):
    values_df = df[column_to_split].str.split(split, expand=True)
    all_features = []
    all_features = list(set(values_df.values.flatten().tolist()))
    all_features.remove(None)
    for feat in all_features:
        df.loc[df['Car Features'].str.contains(feat),feat] = 1
    df = df.fillna(0)
    all
    return df, all_features

### study.optimize(lambda trial: objective(trial,...))
### lgb_params = {**study.best_params, ...}