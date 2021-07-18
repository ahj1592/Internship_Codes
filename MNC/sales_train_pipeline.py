import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import sys, os
from shutil import copyfile
from sklearn.preprocessing import OrdinalEncoder
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (30, 8)
plt.rcParams['axes.grid'] = True
sns.set_theme(rc={"font.family": 'NanumGothic',
                 "axes.unicode_minus": False})
import lightgbm as lgb
import optuna
from collections import defaultdict
import shap
import warnings
warnings.filterwarnings(action='ignore')


def get_parameters(config) -> dict:
    '''
    # set lightGBM parameters by CONFIG
    # Args:
        config (dict): configuration from YAML file
    # Returns:
        params (dict): lightGBM parameters
    '''
    
    params = {}
    for k, v in config['TRAIN']['parameters'].items():
        params[k] = v
    params['seed'] = config['TRAIN']['seed']
    
    return params


def train_model(config, params, evals_result, path, **dataset):
    '''
    # training lightGBM Booster Model
    # Args:
        config (dict): configuration from YAML file
        params (dict): lightGBM parameters
        evals_result (dict): evaluations while training
        path (str): directory or path to save the results
        dataset (dict): kwargs related datasets
            X_train (pandas.DataFrame): train set
            y_train (array-like): labels of train set
            X_valid (pandas.DataFrame): valid set
            y_valid (array-like): labels of valid set
            CAT_FEATURES (arrray-like): categorical columns of df_train
    # Returns:
        params (dict): lightGBM parameters
        model (lightGBM.Booster): booster model
        df_loss (pandas.DataFrame): dataframe contains the losses of trainset, validset
    '''
    
    # ===== HYPERPARAMETER TUNING BY OPTUNA =====
    if config['TRAIN']['optuna']['use']:
        N_TRIALS = config['TRAIN']['optuna']['trials']
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, params, evals_result=evals_result, path=RESULT_PATH, **dataset),
                      n_trials=N_TRIALS, callbacks=[callback_study])
        #-- get best parameters, model and loss
        params.update(study.best_trial.params)
        model = study.user_attrs['best_model']
        df_loss = study.user_attrs['best_loss']
    
    # ===== TRAINING WITHOUT OPTUNA =====
    else:
        train_data = lgb.Dataset(dataset['X_train'],
                                 label=dataset['y_train'],
                                 categorical_feature=dataset['CAT_FEATURES'],
                                 free_raw_data=False)
        valid_data = lgb.Dataset(dataset['X_valid'],
                                 dataset['y_valid'],
                                 categorical_feature=dataset['CAT_FEATURES'],
                                 free_raw_data=False)
        model = lgb.train(params,
                          train_set = train_data,
                          valid_sets = [train_data, valid_data],
                          valid_names = ['train', 'valid'],
                          evals_result = evals_result,
                          verbose_eval=100,
                          categorical_feature=dataset['CAT_FEATURES'])
        df_loss = pd.DataFrame({key: evals_result[key][params['metric']] for key in evals_result.keys()})
    
    return params, model, df_loss


def save_loss(df_loss, path) -> None:
    '''
    # save the loss graph
    # Args:
        df_loss (pandas.DataFrame): dataframe contains the losses of trainset, validset
        path (str): directory or path to save the results
    # Returns:
        None
    '''
    
    sns.lineplot(data=df_loss, x=df_loss.index, y='train', label='train', color='#5392cd')
    sns.lineplot(data=df_loss, x=df_loss.index, y='valid', label='valid', color='#dd8452')
    plt.title('loss', fontsize=30)
    make_single_directory(f'{path}')
    plt.savefig(f'{path}/loss.png', dpi=300)
    plt.clf()
    df_loss.to_csv(f'{path}/loss.csv', index=False)


if __name__ == '__main__':
    # ===== PROJECT DIRECTORY PATH =====
    PRJ_DIR = Path.cwd()
    sys.path.append(str(PRJ_DIR))
    
    # ===== CUSTOMIZED FUNCTIONS =====
    from modules.utils import *
    from modules.metrics import *
    from modules.tuner import *
    from modules.preprocess_data import *
    from modules.visualization import *

    # ===== READ CONFIG & DEFINE DATA PATH =====
    CONFIG_PATH = PRJ_DIR / 'config/sales_train_pipeline.yml'
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    SEED = config['TRAIN']['seed']
    RAW_DATA_DIR_PATH = PRJ_DIR / 'data/train/raw_data' / config['PREPROCESSING']['raw_data_dir']
    PRODUCTCODE_PATH = PRJ_DIR / 'data/train/raw_data' / (config['PREPROCESSING']['kss_productcode_filename']+'.ftr')
    SUBGROUP_PATH = PRJ_DIR / 'data/train/raw_data' / (config['PREPROCESSING']['spec_subgroup_filename']+'.pkl')
    ECON_EFFECT_PATH = PRJ_DIR / 'data/train/raw_data' / (config['PREPROCESSING']['economy_effect_filename']+'.pkl')
    KEYWORD_TREND_PATH = PRJ_DIR / 'data/train/raw_data' / (config['PREPROCESSING']['keyword_trend_filename']+'.pkl')
    SPLIT_TEST_FLAG = config['PREPROCESSING']['split_test_flag']
    SN = generate_serial_number()
    ORDINAL_ENCODER_PATH = PRJ_DIR / f'data/train/{SN}/ordinal_encoder.npy'
    TEST_DATA_DIR_PATH = PRJ_DIR / f'data/train/{SN}/test_data'
    TRAIN_DATA_DIR_PATH = PRJ_DIR / f'data/train/{SN}/split'
    SAVE_DIR_PATH = PRJ_DIR / 'results/train'
    
    os.makedirs(TEST_DATA_DIR_PATH, exist_ok=True)
    os.makedirs(TRAIN_DATA_DIR_PATH, exist_ok=True)

    
    # ===== STEP: PREPROCESSING =====
    #-- load raw data
    df_raw = load_raw_data(RAW_DATA_DIR_PATH)
    #-- cleansing data
    df_raw = cleansing_data(df_raw, PRODUCTCODE_PATH)
    #-- feature engineering
    #-- make sales related columns
    df_part1 = make_sales_columns(df_raw)
    #-- make promotion execute column
    df_part2 = make_promo_execute_column(df_raw)
    #-- make date column
    df_part3 = make_date_column(df_raw)
    #-- make discount rate column
    df_part4 = make_discount_rate_column(df_raw)
    #-- make promotion ordinal number column
    df_part5 = make_promo_ordinal_num_column(df_raw)
    #-- make rental price column
    df_part6 = make_rental_price_column(df_raw)
    #-- make service price column
    df_part7 = make_service_price_column(df_raw)
    #-- merge df_parts
    df_parts = [df_part1, df_part2, df_part3, df_part4, df_part5, df_part6, df_part7]
    part_col = {1: 'promo_execute', 2: 'date', 3: 'discount_rate', 4: 'promo_ordinal_num', 5: '월렌탈료', 6: '서비스료'}
    data_prepared = pd.concat([df_parts[0]]+[df_parts[i][col] for i, col in part_col.items()], axis=1)
    #-- make other columns
    data_prepared = make_other_columns(df_raw, data_prepared, SUBGROUP_PATH, ECON_EFFECT_PATH, KEYWORD_TREND_PATH)
    #-- remove weak sales
    data_prepared = remove_weak_sales(data_prepared)
    #-- make target column
    data_prepared = make_target_column(data_prepared)
    #-- split data into train, valid, test
    train_idx, valid_idx, train_block_idx, valid_block_idx, df_train = split_data(data_prepared, SPLIT_TEST_FLAG, ORDINAL_ENCODER_PATH, TEST_DATA_DIR_PATH, TRAIN_DATA_DIR_PATH)

    
    # ===== STEP: TRAIN =====
    #-- set model name
    model_name = f'LGBM_{SN}'
    print(f'Model name: {model_name}')
    RESULT_PATH = SAVE_DIR_PATH / model_name
    os.makedirs(RESULT_PATH, exist_ok=True)
    #-- save the config
    DATA_ROOT = PRJ_DIR / f'data/train/{SN}'
    copyfile(CONFIG_PATH, f'{DATA_ROOT}/config_copy.yml')
    copyfile(CONFIG_PATH, f'{RESULT_PATH}/config_copy.yml')
    
    #-- set lightgbm parameters
    params = get_parameters(config)
    #-- train the model
    evals_result = {}
    fix_seeds(seed=SEED)
    # X, y of trainset, validset
    TARGET = 'sales'
    FEATURES = ['판매코드', '프로모션유형', '1+1/재렌탈', '할인구분', '의무사용기간', 'BS주기', '할인유형',\
                '상품유형', '용도구분', 'moving_std_28', 'moving_avg_14', 'moving_avg_28', 'sales_sum_7d',\
                'moving_avg_21', 'sales_sum_6d', 'month', 'moving_std_21', 'sales_sum_5d', 'moving_std_7',\
                'promo_execute', 'moving_std_14', 'weekofmonth', 'discount_rate', 'moving_std_5',\
                '정수기_econ_indicator', '정수기_서브그룹', '공기청정기_서브구룹', '판매자수']
    # categorical features
    CAT_FEATURES = ['상품유형', '판매코드', 'BS주기', '용도구분', '의무사용기간', '할인구분', '할인유형',\
                    '1+1/재렌탈', '프로모션유형', 'promo_execute', '정수기_서브그룹', '공기청정기_서브구룹',\
                    'month', 'weekofmonth']
    X_train = df_train.loc[train_idx, FEATURES]
    y_train = df_train.loc[train_idx, TARGET]
    X_valid = df_train.loc[valid_idx, FEATURES]
    y_valid = df_train.loc[valid_idx, TARGET]
    dataset = {'X_train': X_train,
               'y_train': y_train,
               'X_valid': X_valid,
               'y_valid': y_valid,
               'CAT_FEATURES': CAT_FEATURES}
    
    print('****  training  ****')
    params, model, df_loss = train_model(config, params, evals_result, RESULT_PATH, **dataset)
    print('Final Parameters:')
    for k, v in params.items():
        print(f'\t{k}: {v}')
    #-- save parameters, model, loss
    save_object(params, str(RESULT_PATH), 'LGBM_params')
    save_model(model, f'{RESULT_PATH}', model_name)
    save_loss(df_loss, f'{RESULT_PATH}/loss')
    
    
    # ===== STEP: SAVE TRAIN RESULTS =====
    #-- save feature importance(split, gain), SHAP
    make_single_directory(f'{RESULT_PATH}/feature_importance')
    print(f'****  Feature Importance  ****')
    print('\t1. Feature Importance: SPLIT')
    ax = lgb.plot_importance(model, max_num_features=len(FEATURES), importance_type='split')
    ax.set(title=f'Feature Importance (split)',
           xlabel='Feature Importance',
           ylabel='Features')
    ax.figure.savefig(f'{RESULT_PATH}/feature_importance/feature_importance (split).png', dpi=300)
    plt.clf()
    print('\t2. Feature Importance: GAIN')
    ax = lgb.plot_importance(model, max_num_features=len(FEATURES), importance_type='gain')
    ax.set(title=f'Feature Importance (gain)',
           xlabel='Feature Importance',
           ylabel='Features')
    ax.figure.savefig(f'{RESULT_PATH}/feature_importance/feature_importance (gain).png', dpi=300)
    plt.clf()
    print('\t3. SHAP (This step needs some time)')
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_valid)
    fig = shap.summary_plot(shap_values, X_valid)
    plt.savefig(f'{RESULT_PATH}/feature_importance/shap.png', dpi=300, bbox_inches="tight")
    plt.clf()
    
    #-- save metric results
    # predict trainset, validset, (testset)
    pred_train = model.predict(df_train.loc[train_idx, FEATURES])
    true_train = df_train.loc[train_idx, TARGET].reset_index()
    pred_valid = model.predict(df_train.loc[valid_idx, FEATURES])
    true_valid = df_train.loc[valid_idx, TARGET].reset_index()
    if SPLIT_TEST_FLAG:
        df_test = pd.read_pickle(TEST_DATA_DIR_PATH / 'test_data.pkl')
        test_idx = df_test.index
        pred_test = model.predict(df_test.loc[test_idx, FEATURES])
        true_test = df_test.loc[test_idx, TARGET].reset_index()
        
    results_train = pd.concat([true_train, 
                               pd.Series(pred_train, name='pred_sales'),
                               df_train.loc[train_idx, 'sales_day'].reset_index(drop=True)], 
                               axis=1).set_index('index')
    results_valid = pd.concat([true_valid, 
                               pd.Series(pred_valid, name='pred_sales'),
                               df_train.loc[valid_idx, 'sales_day'].reset_index(drop=True)], 
                               axis=1).set_index('index')
    if SPLIT_TEST_FLAG:
        results_test = pd.concat([true_test, 
                                  pd.Series(pred_test, name='pred_sales'),
                                  df_test.loc[test_idx, 'sales_day'].reset_index(drop=True)], 
                                  axis=1).set_index('index')
        
    #-- total summary
    make_single_directory(f'{RESULT_PATH}/summary')
    # concat results of train and valid
    results_train_valid = pd.concat([results_train, results_valid], axis=0).sort_values('index')
    # compute total metrics (MAE, RMSE, MAPE)
    MAE_train = MAE(results_train['sales'], results_train['pred_sales'])
    MAE_valid = MAE(results_valid['sales'], results_valid['pred_sales'])
    RMSE_train = RMSE(results_train['sales'], results_train['pred_sales'])
    RMSE_valid = RMSE(results_valid['sales'], results_valid['pred_sales'])
    MAPE_train = MAPE(results_train['sales'], results_train['pred_sales'])
    MAPE_valid = MAPE(results_valid['sales'], results_valid['pred_sales'])
    if SPLIT_TEST_FLAG:
        MAE_test = MAE(results_test['sales'], results_test['pred_sales'])
        RMSE_test = RMSE(results_test['sales'], results_test['pred_sales'])
        MAPE_test = MAPE(results_test['sales'], results_test['pred_sales'])
    metric_df = pd.DataFrame([['train', MAE_train, RMSE_train, MAPE_train],
                              ['valid', MAE_valid, RMSE_valid, MAPE_valid],],
                             columns=['metric', 'MAE', 'RMSE', 'MAPE'])
    if SPLIT_TEST_FLAG:
        metric_df = pd.DataFrame([['train', MAE_train, RMSE_train, MAPE_train],
                                  ['valid', MAE_valid, RMSE_valid, MAPE_valid],
                                  ['test',  MAE_test,  RMSE_test,  MAPE_test]],
                                 columns=['metric', 'MAE', 'RMSE', 'MAPE'])
    metric_df = metric_df.set_index('metric')
    # save as CSV
    metric_df.to_csv(f'{RESULT_PATH}/summary/metric_summary.csv', encoding='utf-8-sig')
    
    #-- block summary
    # concat ID and results
    df_id_train = df_train.iloc[train_idx].ID
    df_id_valid = df_train.iloc[valid_idx].ID
    df_id_sales_train = pd.concat([df_id_train, results_train], axis=1)
    df_id_sales_valid = pd.concat([df_id_valid, results_valid], axis=1)
    if SPLIT_TEST_FLAG:
        df_id_test = df_test.iloc[test_idx].ID
        df_id_sales_test = pd.concat([df_id_test, results_test], axis=1)
    # IDs are already sorted at preprocessing
    print('**** Compute metrics for TRAIN set ****')
    MAE_trains, RMSE_trains, MAPE_trains = [], [], []
    for _ID_ in tqdm(df_id_sales_train.ID.unique()):
        cur = df_id_sales_train[df_id_sales_train.ID == _ID_]
        MAE_trains.append(MAE(cur.sales, cur.pred_sales))
        RMSE_trains.append(RMSE(cur.sales, cur.pred_sales))
        MAPE_trains.append(MAPE(cur.sales, cur.pred_sales))
    print('**** Compute metrics for VALID set ****')
    MAE_valids, RMSE_valids, MAPE_valids = [], [], []
    for _ID_ in tqdm(df_id_sales_valid.ID.unique()):
        cur = df_id_sales_valid[df_id_sales_valid.ID == _ID_]
        MAE_valids.append(MAE(cur.sales, cur.pred_sales))
        RMSE_valids.append(RMSE(cur.sales, cur.pred_sales))
        MAPE_valids.append(MAPE(cur.sales, cur.pred_sales))
    if SPLIT_TEST_FLAG:
        print('**** Compute metrics for TEST set ****')
        MAE_tests, RMSE_tests, MAPE_tests = [], [], []
        for _ID_ in tqdm(df_id_sales_test.ID.unique()):
            cur = df_id_sales_test[df_id_sales_test.ID == _ID_]
            MAE_tests.append(MAE(cur.sales, cur.pred_sales))
            RMSE_tests.append(RMSE(cur.sales, cur.pred_sales))
            MAPE_tests.append(MAPE(cur.sales, cur.pred_sales))
    # each dataframe has 4 columns: ID, MAE, RMSE, MAPE
    metric_train = pd.DataFrame({'ID': df_id_sales_train.ID.unique(),
                                 'MAE_train': MAE_trains,
                                 'RMSE_train': RMSE_trains,
                                 'MAPE_train': MAPE_trains})
    metric_valid = pd.DataFrame({'ID': df_id_sales_valid.ID.unique(),
                                 'MAE_valid': MAE_valids,
                                 'RMSE_valid': RMSE_valids,
                                 'MAPE_valid': MAPE_valids})
    if SPLIT_TEST_FLAG:
        metric_test = pd.DataFrame({'ID': df_id_sales_test.ID.unique(),
                                    'MAE_test': MAE_tests,
                                    'RMSE_test': RMSE_tests,
                                    'MAPE_test': MAPE_tests})
    # concate dataframe without redundant columns. In this case, remove 'ID' column
    metric_total = pd.concat([metric_train, metric_valid[metric_valid.columns.difference(metric_train.columns)]], axis=1)
    if SPLIT_TEST_FLAG:
        metric_total = pd.merge(metric_total, metric_test, on='ID', how='outer')
    # concat train-valid ID
    df_id_sales_train_valid = pd.concat([df_id_sales_train, df_id_sales_valid], axis=0)
    # use only ID, saels_day; remove other columns
    df_id_sales_train_valid = df_id_sales_train_valid[['ID', 'sales_day']]
    if SPLIT_TEST_FLAG:
        df_id_sales_test = df_id_sales_test[['ID', 'sales_day']]
    df_id_sales_total = df_id_sales_train_valid.copy()
    if SPLIT_TEST_FLAG:
        df_id_sales_total = pd.concat([df_id_sales_train_valid, df_id_sales_test], axis=0).reset_index(drop=True)
    # aggregate by sum
    df_id_sales_sum = df_id_sales_total.groupby('ID').agg({'sales_day': 'sum'}).reset_index()
    # outer join metric results and sales
    block_summary = pd.merge(df_id_sales_sum, metric_total, on='ID', how='outer')
    block_summary = block_summary.rename(columns={'sales_day': 'sales'})
    # ID별 sales_14_sum_mean 구하기
    block_summary['train_sales_14_sum_mean'] = np.nan
    block_summary['valid_sales_14_sum_mean'] = np.nan
    if SPLIT_TEST_FLAG:
        block_summary['test_sales_14_sum_mean'] = np.nan
    block_idx_ID = {}
    for i in block_summary.index:
        block_idx_ID[i] = block_summary.at[i, 'ID']
    # mapping ID -> block_index, (key, value): (ID, block_index), block_index is array-like
    ID_to_train_block_idx = {}
    ID_to_valid_block_idx = {}
    if SPLIT_TEST_FLAG:
        ID_to_test_idx = defaultdict(list)
    for block_idx in train_block_idx:
        _ID_ = df_train.loc[block_idx[0]]['ID']
        ID_to_train_block_idx[_ID_] = block_idx
    for block_idx in valid_block_idx:
        _ID_ = df_train.loc[block_idx[0]]['ID']
        ID_to_valid_block_idx[_ID_] = block_idx
    if SPLIT_TEST_FLAG:
        for i in df_test.index:
            _ID_ = df_test.at[i, 'ID']
            ID_to_test_idx[_ID_].append(i)
    # save TRUE values by IDs
    for row in block_summary.index:
        cur_ID = block_summary.at[row, 'ID']
        x_train = ID_to_train_block_idx[cur_ID]
        y_train = results_train.loc[x_train, 'sales']
        x_valid = ID_to_valid_block_idx[cur_ID]
        y_valid = results_valid.loc[x_valid]['sales']
        block_summary.at[row, 'train_sales_14_sum_mean'] = round(y_train.mean(), 1)
        block_summary.at[row, 'valid_sales_14_sum_mean'] = round(y_valid.mean(), 1)
        if SPLIT_TEST_FLAG:
            x_test = np.array(ID_to_test_idx[cur_ID])
            y_test = results_test.loc[x_test]['sales']
            if len(y_test) != 0: 
                block_summary.at[row, 'test_sales_14_sum_mean'] = round(y_test.mean(), 1)
    # change the columns order
    column_order = ['ID', 'sales', 'train_sales_14_sum_mean', 'valid_sales_14_sum_mean',
                    'MAE_train', 'MAE_valid', 'RMSE_train', 'RMSE_valid',
                    'MAPE_train', 'MAPE_valid']
    if SPLIT_TEST_FLAG:
        column_order = ['ID', 'sales', 'train_sales_14_sum_mean', 'valid_sales_14_sum_mean', 'test_sales_14_sum_mean',
                        'MAE_train', 'MAE_valid', 'MAE_test', 'RMSE_train', 'RMSE_valid', 'RMSE_test',
                        'MAPE_train', 'MAPE_valid', 'MAPE_test']
    block_summary = block_summary[column_order]
    block_summary.to_csv(f'{RESULT_PATH}/summary/block_summary.csv', index=False, encoding='utf-8-sig')

    #-- MAE boxplot
    print('**** Plot MAE error baxplot ****')
    block_summary['판매코드'] = [_ID_.split("-")[1] for _ID_ in block_summary.ID]
    max_error = 50
    metric_types = ['MAE_train', 'MAE_valid']
    if SPLIT_TEST_FLAG:
        metric_types.append('MAE_test')
    plt.rcParams['figure.figsize'] = (60, 8)
    for metric_type in metric_types:
        sns.boxplot(x=block_summary.판매코드, y=block_summary[metric_type], color='tab:blue')
        plt.xticks(rotation=90)
        plt.hlines(y=max_error, xmin=0, xmax=block_summary.판매코드.nunique() - 1, linestyle='--', color='red', linewidth=1)
        plt.grid()
        plt.title(f'{metric_type}', fontsize=30)
        plt.tight_layout()
        make_single_directory(f'{RESULT_PATH}/MAE')
        plt.savefig(f'{RESULT_PATH}/MAE/{metric_type}.png', dpi=300)
        plt.clf()
    
    #-- block plots
    # keyword arguments of PLOT_PREDICT(), PLOT_TRAIN()
    kwargs = {'block_idx_ID': block_idx_ID,
              'ID_to_train_block_idx': ID_to_train_block_idx,
              'ID_to_valid_block_idx': ID_to_valid_block_idx,
              'results_train': results_train,
              'results_valid': results_valid,
              'results_train_valid': results_train_valid,
              'block_summary': block_summary,
              'path': RESULT_PATH}
    common_idx = block_summary[block_summary.ID.isin(df_train.ID.unique())].index
    if SPLIT_TEST_FLAG:
        kwargs['ID_to_test_idx'] = ID_to_test_idx
        kwargs['results_test'] = results_test
        s1 = set(df_train.ID.unique())
        s2 = set(df_test.ID.unique())
        common_ID = list(s1.intersection(s2))
        common_idx = block_summary[block_summary.ID.isin(common_ID)].index
    print('****  plot sales  ****')
    plt.rcParams['figure.figsize'] = (30, 8)
    for block_idx in tqdm(common_idx[-10:]):
            kwargs['idx'] = block_idx
            if SPLIT_TEST_FLAG:
                plot_predict.__globals__.update(kwargs)
                plot_predict(**kwargs)
            else:
                plot_train.__globals__.update(kwargs)
                plot_train(**kwargs)
                
    print('****  All tasks are completed!  ****')