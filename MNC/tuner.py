import numpy as np
import pandas as pd

import lightgbm as lgb
import optuna
from optuna import Trial

from modules.metrics import *
from modules.utils import *

def objective(trial: Trial, lgb_param, df_train, train_idx, valid_idx, features, cat_features, TARGET, evals_result, path, get_model=False):
    '''
    optuna objective function
    
    Args
        trial: trial object
        lgb_param (dict): base parameters of lightGBM model
        df_train (pandas.Dataframe): train set
        train_idx (array-like): indices of train set
        valid_idx (array-like): indices of valid set
        features (array-like): feature columns of df_train
        cat_features (arrray-like): categorical columns of df_train
        TARGET (str): target column of df_train
        evals_result (dict): contains loss dictionary while training
        path (str): path or directory to save model, parameters, loss
        get_model (bool): option for saving model
        
    Returns
        score (float): metric score of current model
        model (lightgbm.Booster): get_model=True
    '''
   
    # randomly select the range of hyperparameters to tune 
    lgb_param['lambda_l1'] = trial.suggest_loguniform('lambda_l1', 1e-8, 1e-1)
    lgb_param['lambda_l2'] = trial.suggest_loguniform('lambda_l2', 1e-8, 1e-1)
    lgb_param['path_smooth'] = trial.suggest_loguniform('path_smooth', 1e-8, 1e-3)
    lgb_param['num_leaves'] = trial.suggest_int('num_leaves', 30, 200)
    lgb_param['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 100)
    lgb_param['max_bin'] = trial.suggest_int('max_bin', 100, 255)
    lgb_param['feature_fraction'] = trial.suggest_uniform('feature_fraction', 0.5, 0.9)
    lgb_param['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.5, 0.9)
    
    
    # set train, valid set
    train_data = lgb.Dataset(df_train.loc[train_idx, features], 
                     label=df_train.loc[train_idx, TARGET])
    valid_data = lgb.Dataset(df_train.loc[valid_idx, features], 
                             label=df_train.loc[valid_idx, TARGET])
    
    # train the model
    model = lgb.train(lgb_param,
                      train_set=train_data,
                      valid_sets = [train_data, valid_data],
                      valid_names = ['train', 'valid'],
                      categorical_feature=cat_features,
                      evals_result=evals_result,
                      verbose_eval=100,
                     )
   
    # ----- set current model
    trial.set_user_attr(key='best_model', value=model)
    
    # ----- save current trial model, parameters, loss
    make_single_directory(f'{path}trials')
    save_model(model, f'{path}trials/', f'model_{trial.number}')
    save_object(lgb_param, f'{path}trials/', f'params_{trial.number}')
    loss_df = pd.DataFrame({
            key: evals_result[key][lgb_param['metric']]
            for key in evals_result.keys()
    })
    loss_df.to_csv(f'{path}trials/loss_{trial.number}.csv', index=False)
    
    
    # p_valid: predicted value of valid set
    # y_valid: true value of valid set
    p_valid = model.predict(df_train.loc[valid_idx, features], 
                            num_iteration=model.best_iteration)
    y_valid = df_train.loc[valid_idx, TARGET]
    
    # ----- get score
    score = MAE(y_valid, p_valid)
    print(f'bset score: {model.best_score}')
    
    if get_model:
        return model
    else:
        return score
    
    
    
def callback_study(study, trial) -> None:
    '''
    save trial's model
    
    Args:
        study
        trial
    
    Returns: None
    '''
    if study.best_trial.number == trial.number:
        study.set_user_attr(key='best_model', value=trial.user_attrs['best_model'])
        