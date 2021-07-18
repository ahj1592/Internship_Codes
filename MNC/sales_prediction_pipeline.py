import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import sys
from sklearn.preprocessing import OrdinalEncoder
import yaml
from shutil import copyfile
import warnings
warnings.filterwarnings(action='ignore')


def replace_code(row):
    """
    # 대체여부를 확인하여 신제품을 기존의 유사한 판매코드로 치환함
    # Args:
        row: 현업 요청 데이터(상품유형, 판매코드만 있는 데이터)의 특정 row
    # Returns:
        row: 대체_판매코드가 채워진 row
    """
    
    global df_similarity, train_code_list
    
    row['모델번호'] = df_similarity.loc[df_similarity['판매코드']==row['판매코드'], '모델번호'].values[0]
    
    if row['대체여부'] == 'N':
        row['대체_모델번호'] = row['모델번호']
        return row
    
    code_list = df_similarity.loc[df_similarity['판매코드']==row['판매코드'], ['판매코드_top1', '판매코드_top2', '판매코드_top3', '판매코드_top4']].values.tolist()[0]
    modelnm_list = df_similarity.loc[df_similarity['판매코드']==row['판매코드'], ['모델번호_top1', '모델번호_top2', '모델번호_top3', '모델번호_top4']].values.tolist()[0]
    try:
        for code, modelnm in zip(code_list, modelnm_list):
            if code in train_code_list:
                row['대체_판매코드'] = code
                row['대체_모델번호'] = modelnm
                return row
        else:
            raise Exception(f'**** 판매코드 {row["판매코드"]}을 대체할 수 있는 판매코드가 없습니다. 상품스펙 데이터를 업데이트 해주세요. ****')
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    # ===== PROJECT DIRECTORY PATH =====
    PRJ_DIR = Path.cwd()
    sys.path.append(str(PRJ_DIR))

    # ===== CUSTOMIZED FUNCTIONS =====
    from modules.utils import *
    from modules.preprocess_data import *

    # ===== READ CONFIG & DEFINE DATA PATH =====
    CONFIG_PATH = PRJ_DIR / 'config/sales_prediction_pipeline.yml'
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    INPUT_FILE_PATH = PRJ_DIR / 'data/prediction' / (config['PREPROCESSING']['input_file_name']+'.xlsx')
    SIMILARITY_PATH = PRJ_DIR / 'data/prediction/similarity/similarity_result.csv'
    TRAIN_DATA_PATH = PRJ_DIR / 'data/train' / config['PREPROCESSING']['train_data_dir'] / 'split/train_data.pkl'
    ORDINAL_ENCODER_PATH = PRJ_DIR / 'data/train' / config['PREPROCESSING']['train_data_dir'] / 'ordinal_encoder.npy'
    RAW_DATA_PATH = PRJ_DIR / 'data/prediction/raw_data' / (config['PREPROCESSING']['raw_data']+'.csv')
    PRODUCTCODE_PATH = PRJ_DIR / 'data/prediction/raw_data' / (config['PREPROCESSING']['kss_productcode_filename']+'.ftr')
    SUBGROUP_PATH = PRJ_DIR / 'data/prediction/raw_data' / (config['PREPROCESSING']['spec_subgroup_filename']+'.pkl')
    ECON_EFFECT_PATH = PRJ_DIR / 'data/prediction/raw_data' / (config['PREPROCESSING']['economy_effect_filename']+'.pkl')
    KEYWORD_TREND_PATH = PRJ_DIR / 'data/prediction/raw_data' / (config['PREPROCESSING']['keyword_trend_filename']+'.pkl')
    PREDICTION_DATE = config['PREPROCESSING']['prediction_date']
    MODEL_DIR_PATH = PRJ_DIR / 'results/train' / config['PREDICTION']['model_name']
    RESULT_SAVE_PATH = PRJ_DIR / 'results/prediction' / (config['PREDICTION']['output_file_name']+'.xlsx')
    #-- save the config
    copyfile(CONFIG_PATH, PRJ_DIR / 'results/prediction' / (config['PREDICTION']['output_file_name']+'_config.yml'))

    # ===== DEFINE PROMOTIN IDS COLUMNS =====
    ids = ['상품유형', '판매코드', 'BS주기', '용도구분', '의무사용기간']
    influx_ids = ['할인구분', '할인유형', '1+1/재렌탈']
    promotion_ids = ['프로모션유형']
    
    
    # ===== READ DATA =====
    df_origin = pd.read_excel(INPUT_FILE_PATH)
    df_origin = df_origin.astype('str')
    df_similarity = pd.read_csv(SIMILARITY_PATH, encoding='utf-8-sig')
    df_similarity = df_similarity.astype('str')
    df_train = pd.read_pickle(TRAIN_DATA_PATH)
    oe = np.load(ORDINAL_ENCODER_PATH, allow_pickle=True).tolist()
    df_raw = pd.read_csv(RAW_DATA_PATH, encoding='cp949')


    # ===== STEP: SIMILARITY =====
    print('****  process similarity step  ****')
    #-- replace new product code
    encoding_list = ids + influx_ids + promotion_ids + ['정수기_서브그룹', '공기청정기_서브구룹', 'month', 'dayofweek', 'weekofmonth']
    df_train[encoding_list] = oe.inverse_transform(df_train[encoding_list])
    train_code_list = list(df_train['판매코드'].unique())
    df_origin['모델번호'] = ''
    df_origin['대체여부'] = 'N'
    df_origin.loc[~df_origin['판매코드'].isin(train_code_list), '대체여부'] = 'Y'
    df_origin['대체_판매코드'] = ''
    df_origin.loc[df_origin['대체여부'] == 'N', '대체_판매코드'] = df_origin['판매코드']
    df_origin['대체_모델번호'] = ''
    df_origin = df_origin.apply(replace_code, axis=1, result_type='broadcast')
    #-- add promotion columns
    df_promo = df_train.loc[df_train['판매코드'].isin(df_origin['대체_판매코드']), ids+influx_ids+promotion_ids].drop_duplicates()
    df_origin = pd.merge(df_origin, df_promo, left_on=['상품유형', '대체_판매코드'], right_on=['상품유형', '판매코드'], how='left')
    df_origin = df_origin.drop(['판매코드_y'], axis=1)
    df_origin = df_origin.rename(columns={'판매코드_x': '판매코드'})
    df_origin = df_origin.sort_values(by=ids+influx_ids+promotion_ids).reset_index(drop=True)
    #-- ID column 생성
    tmp = []
    for nm in (['상품유형', '대체_판매코드', 'BS주기', '용도구분', '의무사용기간']+influx_ids+promotion_ids)[1:]:
        tmp.append(df_origin[f'{nm}'].astype('str'))
    else:
        df_origin['ID'] = df_origin[ids[0]].str.cat(tmp, sep='-')
    print('****  similarity step complete!!  ****')

    
    # ===== STEP: PREPROCESSING =====
    print('****  process preprocessing step  ****')
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
    #-- filtering
    data_prepared = data_prepared[data_prepared['date'] == str(PREDICTION_DATE)]
    data_prepared = data_prepared[data_prepared['ID'].isin(list(df_origin.ID.unique()))]
    data_prepared = data_prepared.sort_values(by=ids+influx_ids+promotion_ids)
    data_prepared = data_prepared.reset_index(drop=True)
    #-- encoding
    encoding_list = ids + influx_ids + promotion_ids + ['정수기_서브그룹', '공기청정기_서브구룹', 'month', 'dayofweek', 'weekofmonth']
    data_prepared[encoding_list] = oe.transform(data_prepared[encoding_list])
    print('****  preprocessing step complete!!  ****')


    # ===== STEP: PREDICTION =====
    print('****  process prediction step  ****')
    TARGET = 'sales'
    FEATURES = ['판매코드', '프로모션유형', '1+1/재렌탈', '할인구분', '의무사용기간', 'BS주기', '할인유형', '상품유형', '용도구분',\
                'moving_std_28', 'moving_avg_14', 'moving_avg_28', 'sales_sum_7d', 'moving_avg_21', 'sales_sum_6d', 'month',\
                'moving_std_21', 'sales_sum_5d', 'moving_std_7', 'promo_execute', 'moving_std_14', 'weekofmonth', 'discount_rate',\
                'moving_std_5', '정수기_econ_indicator', '정수기_서브그룹', '공기청정기_서브구룹', '판매자수']
    model = load_model(str(MODEL_DIR_PATH), config['PREDICTION']['model_name'])
    pred_data_idx = data_prepared.index
    pred_sales = model.predict(data_prepared.loc[pred_data_idx, FEATURES])
    #-- make the result file
    data_prepared = data_prepared.assign(pred_sales=pred_sales)
    data_prepared = data_prepared[['판매코드', 'ID', 'pred_sales', '월렌탈료', '의무사용기간', '서비스료']]
    data_prepared['매출'] = data_prepared['월렌탈료'] * data_prepared['의무사용기간']
    data_prepared['금융리스_매출'] = (data_prepared['월렌탈료'] - data_prepared['서비스료']) / 1.1 * data_prepared['의무사용기간']
    data_prepared = data_prepared.drop(['판매코드', '월렌탈료', '의무사용기간', '서비스료'], axis=1)
    df_final = pd.merge(df_origin, data_prepared, on='ID', how='inner')
    df_final = df_final.drop(['ID'], axis=1)
    df_final['rank'] = df_final.groupby('판매코드')['pred_sales'].rank('dense', ascending=False)
    df_final['rank'] = df_final['rank'].astype('int')
    df_final['BS주기'] = df_final['BS주기']+'개월'
    df_final['의무사용기간'] = df_final['의무사용기간']+'개월'
    df_final['할인유형'] = df_final['할인유형']+'년'
    df_final = df_final.replace({'BS주기': {'0': '0-주기없음'},\
                                 '용도구분': {'0': '0-공통', '1': '1-일반', '2': '2-업소', '3': '3-일반', '4': '4-업소', '5': '5-홈케어', '6': '6-특별',\
                                             '7': '7-택배', '9': '9-기타'},\
                                 '할인구분': {'0': '0-없음', '1': '1-법인', '2': '2-직원', '3': '3-중고보상', '5': '5-법인단체', '6': '6-패키지',\
                                             '7': '7-다자녀/다문화/장애인', '8': '8-일반'},\
                                 '1+1/재렌탈': {'없음': '제도할인없음', '02': '02-재렌탈', '03': '03-1+1', '14': '14-패키지2대', '15': '15-패키지3대',\
                                               '16': '16-패키지4대이상', '17': '17-총판특별할인', '18': '18-특별할인(자동1)', '19': '19-5년약정할인',\
                                               '20': '20-특별할인(선택)', '21': '21-에듀제휴할인', '22': '22-제휴업체할인', '23': '23-제휴패키지할인',\
                                               '71': '71-특별할인(자동2)', '81': '81-국고보조(선택)', '82': '82-특별할인(선택2)'},\
                                 '프로모션유형': {'0': '0-프로모션없음', '1': '1-렌탈할인', '2': '2-할인개월', '3': '3-무료개월', '4': '4-사은품',\
                                                '5': '5-렌탈요금', '6': '6-설치월면제+무료개월', '7': '7-매년1차월무료', '8': '8-0차월면제+할인개월'}})
    df_final = df_final.sort_values(by=['상품유형', '판매코드', 'rank'])
    df_final = df_final.reset_index(drop=True)
    df_final = df_final.round({'pred_sales': 2, '매출': 2, '금융리스_매출': 2})
    df_final.to_excel(RESULT_SAVE_PATH, encoding='utf-8-sig', index=False)
    print('****  prediction step complete!!  ****')
    print(f'****  Save path: {RESULT_SAVE_PATH}  ****')