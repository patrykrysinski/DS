import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from functions import objective
class model:
    def __init__(self,numerical_features,cat_features,train_X,obj, train_y,seed,fold_no):
        self.numerical_features = numerical_features
        self.cat_features = cat_features
        self.train_X = train_X
        self.obj = obj 
        self.train_y = train_y 
        self.seed = seed 
        self.fold_no = fold_no
    # def objective(self, trial):
    #     hyperparams = {'objective':trial.suggest_categorical('objective',['regression','regression_l1']),
    #                     'boosting': trial.suggest_categorical('boosting',['gbdt','dart']),
    #                     'learning_rate' : trial.suggest_float('learning_rate',0.05,0.8),
    #                     'num_iterations':trial.suggest_int('num_iterations',10,1000),
    #                     'max_depth': trial.suggest_int('max_depth',2,15),
    #                     'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',10,50),
    #                     'bagging_fraction': trial.suggest_float('bagging_fraction',0.4,1),
    #                     'feature_fraction': trial.suggest_float('bagging_fraction',0.4,1),
    #                     'lambda_l1': trial.suggest_float('lambda_l1',0,10),
    #                     'lambda_l2': trial.suggest_float('lambda_l2',0,10),
    #                     'num_leaves':  trial.suggest_int('num_leaves',5,100),
    #                     'linear_lambda': trial.suggest_float('linear_lambda',0,10),
    #                     'min_gain_to_split': trial.suggest_float('min_gain_to_split',0,10),
    #                     'saved_feature_importance_type': 1,
    #                     'drop_rate':trial.suggest_float('drop_rate',0.05,1),
    #                     'max_drop': trial.suggest_int('max_drop',20,100),
    #                     'seed': self.seed,
    #                     'force_col_wise': True ,
    #                     'make_var': trial.suggest_categorical('make',['make_new_1_encoded','make_new_2_encoded']) }
    #     if ('make_new_1_encoded' in self.cat_features) &  ('make_new_2_encoded' in self.cat_features):
    #         self.cat_features.remove(hyperparams['make_var'])
    #     model_feature = []
    #     model_feature.extend(self.cat_features)
    #     model_feature.extend(self.numerical_features)

    #     train_lgb = lgb.Dataset(data=self.train_x[model_feature], label=self.train_y, categorical_feature=self.cat_features)
    #     model = lgb.cv(params=hyperparams,
    #                 train_set=train_lgb,
    #                 #num_boost_round=num_iterations,
    #                 seed =self.seed,
    #                 nfold=self.fold_no,
    #                 metrics=self.obj
    #     )
        
    #     return model[f'valid {self.obj}-mean'][hyperparams['num_iterations']-1]
    
    def train(self,n_trials):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial,
                                               train_x=self.train_X,
                                               obj=self.obj, 
                                               train_y=self.train_y,
                                               seed=self.seed,
                                               numerical_features=self.numerical_features,
                                               cat_features=self.cat_features), n_trials=n_trials)
        if 'make_var' in study.best_params.keys():
            params = {key: value for key, value in study.best_params.items() if key != 'make_var'}
            if (study.best_params['make_var'] =='make_new_2_encoded') & ('make_new_1_encoded' in self.cat_features):
                self.cat_features.remove('make_new_1_encoded')
            elif (study.best_params['make_var'] =='make_new_1_encoded') & ('make_new_2_encoded' in self.cat_features):
                self.cat_features.remove('make_new_2_encoded')
        else:
            params = study.best_params
        self.final_models_feature = []
        self.final_models_feature.extend(self.cat_features)
        self.final_models_feature.extend(self.numerical_features)

        train_lgb = lgb.Dataset(data=self.train_X[self.final_models_feature], label=self.train_y, categorical_feature=self.cat_features)
        m1 = lgb.cv(params=params,
                    train_set=train_lgb,
                   nfold=self.fold_no,
                   seed=self.seed,
                   metrics=self.obj,
                   return_cvbooster=True)
        self.model= m1['cvbooster']
        self.model_results = m1[f'valid {self.obj}-mean']
    def predict(self,df,y_name):
        if len(self.model.boosters)>0:
            pred = self.model.predict(df[self.model.boosters[0].feature_name()])
            pred_df = pd.DataFrame(data=[pred][0]).transpose()
            pred_df.columns = [f'prediction_cv{i}' for i in np.arange(len(pred_df.columns))]
            pred_df['prediction'] = pred_df.mean(axis=1)
            pred_df['y_true'] = df[y_name].reset_index(drop =True)
            return pred_df
        else:
            "There is no model to predict"
    def importance_calculation(self):
        importance_df = pd.DataFrame(index=self.final_models_feature)
        if len(self.model.boosters)>0:
            for i in range(len(self.model.boosters)):
                importance_df[f'imp_{i}'] = self.model.boosters[i].feature_importance()
            importance_df['importance'] = importance_df.median(axis=1)
            importance_df = importance_df.sort_values(by='importance', ascending=False)
        self.importance = importance_df['importance']