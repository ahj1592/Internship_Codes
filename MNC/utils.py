"""공용 함수
    * Directory IO
    * File IO
    * Logger
    * System
    * Seed
    * Plotter

TODO:
    * docstring 작성

NOTES:
    * 

REFERENCE:    
    
UPDATED: 2O21.06.10
"""

import pandas as pd
import numpy as np
import logging
import random
import yaml
import csv
import os
from glob import glob
import uuid
from datetime import datetime
import joblib
import pickle

# ====== Directory IO =====
def generate_serial_number():
    return datetime.today().strftime('%Y-%m-%d')+'-'+str(uuid.uuid1()).split('-')[0]

def make_directory(path, serial_num):
    os.makedirs(f'{path}{serial_num}', exist_ok=True)

def make_single_directory(path):
    os.makedirs(f'{path}', exist_ok=True)

# ====== File IO =====
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def save_yaml(path, obj):
    with open(path, 'w') as f:
        yaml.dump(obj, f, sort_keys=False)
        
def load_excel(data_dir, datafile):
    df = pd.read_excel(f'{data_dir}{datafile}.xlsx') 
    return df

def load_csv(data_dir, datafile):
    df = pd.read_csv(f'{data_dir}{datafile}.csv', encoding='cp949')
    return df

def load_feather(filepath, filename):
    df = pd.read_feather(f'{filepath}{filename}.ftr')
    print(f'File Loaded from {filepath}{filename}.ftr')
    return df
    
def save_feather(df_input, filepath, filename):
    df = df_input.copy()
    files_present = glob(filepath+filename+'.ftr')
    if not files_present:
        df.to_feather(f'{filepath}{filename}.ftr')
        print(f'File Saved to {filepath}{filename}.ftr')
    else:
        print('WARNING: This feather file already exists!')

def load_pickle(filepath, filename):
    df = pd.read_pickle(f'{filepath}{filename}.pkl')
    print(f'File Loaded from {filepath}{filename}.pkl')
    return df

def save_pickle(df_input, filepath, filename):
    df= df_input.copy()
    files_present = glob(filepath+filename+'.pkl')
    if not files_present:
        df.to_pickle(f'{filepath}{filename}.pkl')
        print(f'File Saved to {filepath}{filename}.pkl')
    else:
        print('WARNING: This pickle file already exists!')
        
def save_numpy(nump, filepath, filename):
    files_present = glob(filepath+filename+'.npy')
    if not files_present:
        np.save(filename, nump)
        print(f'File Saved to {filepath}{filename}.npy')
    else:
        print('WARNING: This numpy file already exists!')

def save_w2v(model, filepath, filename):
    files_present = glob(filepath+filename)
    if not files_present:
        model.wv.save_word2vec_format(filepath+filename)
        print(f'File Saved to {filepath}{filename}')
    else:
        print('WARNING: This w2v model file already exists!')

def save_image(self, model):
    filename = f'{self.save_path}{self.data_date}_{self.datasource}_{self.save_version}'
    print('Saving data to: ', filename)

    files_present = glob.glob(filename)
    if not files_present:
        plt.savefig(filename, dpi=300)
        print('Export complete!')
    else:
        print('WARNING: This  image file already exists!')      
        
        
def save_model(model, filepath, filename):
    files_present = glob(filepath+filename+'.pkl')
    if not files_present:
        joblib.dump(model, f'{filepath}{filename}.pkl')
        print(f'Model Saved to {filepath}{filename}.pkl')
    else:
        print('WARNING: This model is already exists!')
        
        
def load_model(filepath, filename):
    if filepath[-1] != '/':
        filepath += '/'
    model = joblib.load(f'{filepath}{filename}.pkl')
    print(f'Model Loaded from {filepath}{filename}.pkl')
    return model

def save_object(item, filepath, filename):
    if filepath[-1] != '/':
        filepath += '/'
    with open(f'{filepath}{filename}', 'wb') as f:
        pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)

def load_object(filepath, filename):
    if filepath[-1] != '/':
        filepath += '/'
    with open(f'{filepath}{filename}', 'rb') as f:
        return pickle.load(f)


# ===== SET SEED =====
def fix_seeds(seed = 42, use_torch=False):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 함
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if use_torch: 
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
        
        
def detach_d(x):
    x = str(x)
    return int(x[2:])