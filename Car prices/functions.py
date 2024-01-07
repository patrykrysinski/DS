from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import os
import plotly.express as px
import numpy as np
import lightgbm as lgb

def kaggle_download(user, dataset):

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(f"{user}/{dataset}")
    zips = [i for i in os.listdir() if i.endswith('zip')]
    for zip in zips:
        zf = ZipFile(zip)
        zf.extractall() 
        zf.close()

def bubble_plot(df,x,y,ids_col,groupper,list_to_change_location,agg_func = np.mean):
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
    
def barplot_share(df, column):
    agg_df = df.groupby(column).count().reset_index().iloc[:,:2]
    agg_df.columns = [column,'number']
    agg_df['share'] =  np.round(agg_df['number']/ agg_df['number'].sum(),3)
    agg_df = agg_df.sort_values(by='share', ascending = False)
    return px.bar(data_frame=agg_df, x=column,y='share',title=f"Share by {column}" )

def objective(trial, train_x, train_y,obj,seed,numerical_features,cat_features,fold_no = 3):
    hyperparams = {'objective':obj,
                    'boosting': trial.suggest_categorical('boosting',['gbdt','rf','dart']),
                    'num_iterations': trial.suggest_int('num_iterations',10,1000),
                    'learning_rate' : trial.suggest_float('learning_rate',0.05,0.8),
                    'max_depth': trial.suggest_int('max_depth',2,15),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',10,50),
                    'bagging_fraction': trial.suggest_float('bagging_fraction',0.4,1),
                    'feature_fraction': trial.suggest_float('bagging_fraction',0.4,1),
                    'lambda_l1': trial.suggest_float('lambda_l1',0,10),
                    'lambda_l2': trial.suggest_float('lambda_l2',0,10),
                    'linear_lambda': trial.suggest_float('linear_lambda',0,10),
                    'min_gain_to_split': trial.suggest_float('min_gain_to_split',0,10),
                    'saved_feature_importance_type': 1,
                    'drop_rate':trial.suggest_float('drop_rate',0.05,1),
                    'max_drop': trial.suggest_int('max_depth',20,100),
                    'seed': seed  }
    if ('make_new_1' in cat_features) &  ('make_new_2' in cat_features):
        make_var = trial.suggest_categorical('make',['make_new_1','make_new_2'])
        cat_features.remove(make_var)
    model_feature = []
    model_feature.extend(cat_features)
    model_feature.extend(numerical_features)
    train_lgb = lgb.Dataset(data=train_x[model_feature], label=train_y, categorical_feature=cat_features)
    model = lgb.cv(params=hyperparams,
                   train_set=train_lgb,
                   seed =seed,
                   nfold=fold_no,
                   metrics=obj
    )
    
    return model[f'valid {obj}-mean'][hyperparams['num_iterations']-1]

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