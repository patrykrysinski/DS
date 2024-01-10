import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from functions import objective
class model:
    def __init__(self,
                 numerical_features : list[str],
                 cat_features: list[str],
                 train_X: pd.DataFrame,
                 obj: str, 
                 train_y: pd.DataFrame,
                 seed:str,
                 fold_no: int):
        
        """ Class model to train and optimize lightgbm model

            Args:
            numerical_features: list of numerical features to the model,
            cat_features : list of categorical features to the model,
            train_X : Data frame included train X's to the model,
            obj: evaluation metric,
            train_y : Data frame included train y to the model,
            seed: random seed,
            fold_no: number of folds in the model.
        """
        
        try:
            t = train_X[numerical_features]
            t = train_X[cat_features]
        except KeyError:
            print("The list ofgiven features don't match columns in train x")
        if fold_no <=1:
            raise ValueError("Number of folds need to be greater than 1")
        
        self.numerical_features = numerical_features
        self.cat_features = cat_features
        self.train_X = train_X
        self.obj = obj 
        self.train_y = train_y 
        self.seed = seed 
        self.fold_no = fold_no
    
    def train(self,n_trials: int):
        """
        Method to train model with optimization

        Args:
        n_trials: number of optuna trials
        """
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial,
                                               train_x=self.train_X,
                                               obj=self.obj, 
                                               train_y=self.train_y,
                                               seed=self.seed,
                                               numerical_features=self.numerical_features,
                                               cat_features=self.cat_features,
                                               ), 
                                               n_trials=n_trials)
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

    def predict(self,df: pd.DataFrame,y_name: str):
        """
        Method to predict built model on a given dataset.
        
        Args:
        df: Data frame to predict
        y_name: Name of target variable
        """

        if y_name not in df.columns:
            raise KeyError(f'{y_name} not in dataframe columns')
        if len(self.model.boosters)>0:
            try:
                df_to_pred = df[self.model.boosters[0].feature_name()]
            except KeyError:
                print("Data frame dosen't contain all X's")
            pred = self.model.predict(df_to_pred)
            pred_df = pd.DataFrame(data=[pred][0]).transpose()
            pred_df.columns = [f'prediction_cv{i}' for i in np.arange(len(pred_df.columns))]
            pred_df['prediction'] = pred_df.mean(axis=1)
            pred_df['y_true'] = df[y_name].reset_index(drop =True)
            return pred_df
        else:
            raise AttributeError("There is no model to predict")
        
    def importance_calculation(self):

        """ Method to calculate model's importance """

        importance_df = pd.DataFrame(index=self.final_models_feature)
        if len(self.model.boosters)>0:
            for i in range(len(self.model.boosters)):
                importance_df[f'imp_{i}'] = self.model.boosters[i].feature_importance()
            importance_df['importance'] = importance_df.median(axis=1)
            importance_df = importance_df.sort_values(by='importance', ascending=False)
            self.importance = importance_df['importance']
        else:
            raise  AttributeError("There is no model to calculate importance")


