import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import mean_absolute_error as MAE
from modules.metrics import *
from modules.utils import make_directory, make_single_directory

import os

def plot_roc(y_trues, y_preds, labels, x_max=1.0, path=None):
    fig, ax = plt.subplots()
    for i, y_pred in enumerate(y_preds):
        y_true = y_trues[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, label='%s; AUC=%.3f' % (labels[i], auc), marker='o', markersize=1)

    ax.legend()
    ax.grid()
    ax.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), linestyle='--')
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_xlim([-0.01, x_max])
    _ = ax.set_ylabel('True Positive Rate')
    
    save_path = f'{path}plots'
    make_single_directory(save_path)
    ax.figure.savefig(f'{save_path}/ROC_curve(Adversarial Validation).png', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.clf()
    

def show_notice(save, path) -> None:
    '''Print Infomations of save mode and path
    
    Args:
        save (bool): save mode
        path (str): path for saving image plot
        
    Returns: None
    
    '''
    
    print('+===========================NOTICE===========================+') # length is 62
    if not save:
        print('{:<60}'.format('| Saving mode is OFF.'), '|')
        print('{:<60}'.format('| If you want to save images from EDA, pass save=True'), '|')
    else:
        print('{:<60}'.format('| Saving mode is ON.'), '|')
    
        if path is None or path == '':
            print('{:<60}'.format('| Images will be saved at current directory.'), '|')
        else:
            print('{:<60}'.format('| Images will be saved at {}.'.format(path)), '|')
    print('+============================================================+')
    
    return



def init_plot_theme() -> None:
    '''Initialize the seaborn theme setting
    
    Args: None
    Returns: None
    
    Note:
        palette ref: http://hleecaster.com/python-seaborn-color/
                     https://seaborn.pydata.org/tutorial/color_palettes.html
        
        한글 폰트 및 새로운 폰트 사용법
        font.family: https://devstarsj.github.io/computer/2018/10/13/jupyter.matplotlib.korean.font/
            1. 사용할 폰트가 있는 위치 확인
            2. 폰트 파일을 matplotlib.matplotlib_fname()의 *./mpl-data/fonts/ttf에 copy&paste
            3. reset the cache matplotlib. matplotlib.get_configdir()가 캐시 위치
            4. delete(or backup) cache file (.json)
            5. restart the kernel
            
    '''
    sns.reset_defaults()
    
    # if there is no such font, then it will be changed automatically (with warning message)
    sns.set_theme(
        font_scale=1,
        #context='paper',
        #style='white',
        palette='Set1',
        rc={"font.size": 10, 
            "axes.titlesize": 16, 
            "axes.labelsize": 10,
            "patch.linewidth": 0,
            "patch.edgecolor":'none',
            "font.family": 'NanumGothic', 
            "axes.unicode_minus": False, #한글 사용시, 마이너스 폰트 깨짐 현상 방지
           }
    )
    return


def show_countplot(input_df, *columns, horizontal=False, vertical=True, save=False, path=None) -> None:
    init_plot_theme()
    '''Draw countplot given dataframe, vertically in default 
       ONLY category types are possible.
    
    Args:
        input_df (pandas.Dataframe): input dataframe
        *columns (str): tuples of column names
        horizontal (bool): option for draw horizontally 
        vertical (bool): option for draw vertially
        save (bool): option for save
        path (str): path for saving image plot
        
    Returns: None
    
    Note:
        If there is no argument for columns, then plot ALL category columns
    '''
   
    show_notice(save=save, path=path)
    
    
    df = input_df.copy()
    
    # if do not pass arguments, uses ALL columns
    if len(columns) == 0:
        columns = df.columns
    
    ignore = [] # list that contains non-category columns
    for col in columns:
        # object dtype is option
        if is_categorical_dtype(df[col]) or df[col].dtype == np.object:
            
            #============================
            #print(col)
            #=============================
            
            # Illustrate horizontal / vertical graph
            if horizontal or not vertical:
                sns.countplot(data=df, y=col)
            else:
                sns.countplot(data=df, x=col)
                
            # append the TITLE
            title = f'The number of {col}'
            title = title.replace('/', '_')
            plt.title(title)
            
            # store the plot image
            if save:
                if path:
                    plt.savefig(f'{path}{title}.png', dpi=300)
                    print(f'Image is saved at {path}{title}.png')
                else:
                    plt.savefig(f'{title}.png', dpi=300)
                    print('Image is saved at current directory.')

            plt.show()
            plt.clf() # clear the plot
        else: # remove non-category columns
            ignore.append(col)
    
    # notice that which columns is not availalbe
    if ignore:
        ignore.sort()
        print(f'The number of non-category columns : {len(ignore)}')
        print(f'Non-category columns (including object): {ignore}')
    return


def show_distplot(**kwargs) -> None:
    init_plot_theme()
    '''Illustrate the distributions of two dataframes with common columns
       2개의 dataframe의 공통 컬럼들에 대해서 분포를 그린다.
    
    Args:
        **train (pandas.DataFrame): train dataset
        **test (pandas.DataFrame): test dataset
        **height (int, float): height of plot
        **save (bool): option for save
        **path (str): path for saving image plot
        
    Returns: None
    '''
    if 'train' not in kwargs:
        raise Exception('You must assign the train dataframe. pass the train=df_name')
    if 'test' not in kwargs:
        raise Exception('You must assign the test dataframe. pass the test=df_name')
        
    # train, test가 dataframe이어야 하는 assert 추가
    
    height = kwargs['height'] if 'height' in kwargs else 5
    save = kwargs['save'] if 'save' in kwargs else False
    path = kwargs['path'] if 'path' in kwargs else None
    
    
    show_notice(save=save, path=path)
        
    
    assert type(kwargs['train']) == pd.core.frame.DataFrame, 'train must be dataframe of pandas.'
    assert type(kwargs['test']) == pd.core.frame.DataFrame, 'test must be dataframe of pandas.'
    
    train_df = kwargs['train'].copy()
    test_df = kwargs['test'].copy()
    
    # COMMON_COLS: TRAIN, TEST dataframe의 공통 columns
    # IGNORE: TRAIN, TEST dataframe의 공통되지 않는 columns
    common_cols = train_df.columns.intersection(test_df.columns)
    ignore = train_df.columns.difference(test_df.columns)
    ignore = test_df.columns.difference(train_df.columns).union(ignore)
    ignore = list(ignore)
    
    # TRAIN과 TEST의 공통 column이 없으면 EDA 진행 불가.
    if len(common_cols) == 0:
        print('There is no common columns. Cannot show the distribution.')
        return
    
    # ID column을 추가 -> TRAIN, TEST 구별하기 위한 column
    train_df['id'] = 'train'
    test_df['id'] = 'test'
    total_df = pd.concat([train_df, test_df])
    for col in common_cols:
        # exclude the non-numeric columns
        if not is_numeric_dtype(total_df[col]):
            ignore.append(col)
            continue
        
        # width = height * aspect
        # illustrate the distribution + KDE graph
        sns.displot(data=total_df, x=col, hue='id', kde=True, height=height, aspect=1)
        
        # set the TITLE
        title = f'Distribution of {col}-Count'
        title = title.replace('/', '_')
        plt.title(title)
        
        # save the image
        if save:
            if path:
                plt.savefig(f'{path}{title}.png', dpi=300)
                print(f'Image is saved at {path}{title}.png')
            else:
                plt.savefig(f'{title}.png', dpi=300)
                print('Image is saved at current directory.')
                    
        plt.show()
        plt.clf() # clean the plot
    
    
    if ignore:
        ignore.sort()
        print(f'The number of non-numeric columns : {len(ignore)}')
        print(f'Not common columns : {ignore}')
    return



def show_corr_heatmap(input_df, figsize=(11, 9), title=None, save=False, path=None) -> None:
    init_plot_theme()
    '''Illustrate the correlation heatmap of given dataframe
    
    Args:
        input_df (pandas.DataFrame): input dataframe
        figsize (2-tuple): figsize of heatmap
        title (str): title of heatmap
        save (bool): option for save
        path (str): path for saving image plot
        
    Returns: None
    
    Note:
        I refered to this site.
        https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    '''
    
    show_notice(save=save, path=path)
    
    df = input_df.copy()
    
    # remove not numeric columns
    ignore = [] # list that contains non-numeric columns
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            ignore.append(col)
            del df[col]
    
    corr = df.corr() # Compute the correlation matrix
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Generate a mask for the upper triangle

    # Set up the matplotlib figure
    plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1., vmax=1., center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    # set the TITLE
    if title is None:
        title = 'Correlation map'
    title = title.replace('/', '_')
    plt.title(title)
    
    if save:
        if path:
            plt.savefig(f'{path}{title}.png', dpi=300)
            print(f'Image is saved at {path}{title}.png')
        else:
            plt.savefig(f'{title}.png', dpi=300)
            print('Image is saved at current directory.')
            
    plt.show()
    plt.clf()
    
    if ignore:
        ignore.sort()
        print(f'The number of non-numeric columns : {len(ignore)}')
        print(f'Non-numeric columns : {ignore}')
    return



def plot_predict(**kwargs):
    #plot_predict.__globals__.update(kwargs)
    
    cur_ID = block_idx_ID[idx]
    
    # train X, y, prediction
    x_train = ID_to_train_block_idx[cur_ID]
    y_train = results_train.loc[x_train, 'sales']
    p_train = results_train.loc[x_train, 'pred_sales']

    # valid X, y, prediction
    x_valid = ID_to_valid_block_idx[cur_ID]
    y_valid = results_valid.loc[x_valid]['sales']
    p_valid = results_valid.loc[x_valid]['pred_sales']

    # train + valid
    x_train_valid = np.sort(np.append(x_train, x_valid))
    y_train_valid = results_train_valid.loc[x_train_valid]['sales']
    p_train_valid = results_train_valid.loc[x_train_valid]['pred_sales']

    # test X, y, prediction
    x_test = np.array(ID_to_test_idx[cur_ID])
    y_test = results_test.loc[x_test]['sales']
    p_test = results_test.loc[x_test]['pred_sales']

    # reset x_axis[0] as 0
    x_total = x_train_valid - x_train_valid[0]
    x_test_ = x_test - x_test[0] + x_total[-1] + 1
    x_total = np.append(x_total, x_test_)
    y_total = np.append(y_train_valid, y_test)
    p_total = np.append(p_train_valid, p_test)

    # make 2 subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    fig.subplots_adjust(hspace=0.05)

    # ===== PLOT SALES 
    # ----- true, train, test   
    sns.lineplot(x=x_total, y=y_total, label='true', color='black', linewidth=0.8, alpha=0.7, ax=ax1)
    sns.lineplot(x=x_train_valid - x_train_valid[0], y=p_train_valid, label='train', color='#5392cd', linewidth=1.1, ax=ax1)
    sns.lineplot(x=x_test_, y=p_test,  color='#dd8452', label='pred', linewidth=1.1, ax=ax1)
    if len(x_test_) >= 14:
        ax1.axvspan(x_test_[0], x_test_[13], facecolor ='#929292', alpha = 0.5)
    else:
        ax1.axvspan(x_test_[0], x_test_[-1], facecolor ='#929292', alpha = 0.5)
    
       
    
    TITLE = f'Block No.{idx:04d}'
    ax1.set_title(TITLE, fontsize=16)
    ax1.legend(prop={'size': 14})
    # rotate xticks
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)

    # ===== TABLE METRIC CONFIGURATION
    column_headers = ('Train', 'Valid', 'Test')
    row_headers = ['Length', 'Mean Sales', 'MAE']

    TEXT = f'Total length: {len(x_total)}(day)'
    # ----- table contents
    cell_text=[
        [f'{len(x_train)} day', f'{len(x_valid)} day', f'{len(x_test_)} day']
        , [f'{y_train.mean():.1f}', f'{y_valid.mean():.1f}', f'{y_test.mean():.1f}']
        , [f"{block_summary.at[idx, 'MAE_train']:.3f}", f"{block_summary.at[idx, 'MAE_valid']:.3f}", f"{block_summary.at[idx, 'MAE_test']:.3f}"]
         ]


    # ----- colors of columns, rows
    ccolors = plt.cm.Greys(np.full(len(column_headers), 0.3))
    rcolors = plt.cm.Greys(np.full(len(row_headers), 0.3))

    # ----- make table
    table = ax2.table(colLabels=column_headers
                      , rowLabels=row_headers
                      , cellText=cell_text
                      , colWidths  = [0.1,0.1,0.1,0.1] 
                      , rowColours=rcolors
                      , colColours=ccolors
                      , cellLoc ='center'
                      , loc='center')

    table.set_fontsize(16)
    table.scale(1,2)
    ax2.axis('tight')
    ax2.axis('off')


    # ===== TEXT for block information =====
    block_info = cur_ID.split('-')
    SALES_INFO = f"Total Sales: {int(block_summary.at[idx, 'sales'])}"
    INFO = f'상품유형: {block_info[0]}, 판매코드: {block_info[1]}\n\
    BS주기: {block_info[2]}, 용도구분: {block_info[3]}, 의무사용기간: {block_info[4]} \n\
    할인구분: {block_info[5]}, 할인유형: {block_info[6]}, 1+1/재렌탈: {block_info[7]}\n\
    프로모션유형: {block_info[8]}'
    
                  #0.15, 0.135
    plt.gcf().text(0.12, 0.18, TEXT + '\n'+ SALES_INFO + '\n' + '\n' + INFO, fontsize=16, horizontalalignment='left')

    # ===== save plot image including TEXT, TABLE
    save_path = f'{path}plots_vline'
    make_single_directory(save_path)
    plt.savefig(f'{save_path}/{TITLE}_{cur_ID}.png', dpi=200, bbox_inches='tight')
    plt.clf()

    # ---- 확대 부분
    fig_part, ax_part = plt.subplots()
    x_train_part = x_train_valid - x_train_valid[0]
    
    for i in range(len(x_total)):
        if x_total[i] == x_train_part[-14]:
            tmp = i
            break
    
    sns.lineplot(x=x_total[tmp:], y=y_total[tmp:], label='true', color='black', linewidth=0.8, alpha=0.7, ax=ax_part)
    sns.lineplot(x=x_train_part[-14:], y=p_train_valid[-14:], label='train', color='#5392cd', linewidth=1.1, ax=ax_part)
    sns.lineplot(x=x_test_, y=p_test,  color='#dd8452', label='pred', linewidth=1.1, ax=ax_part)
    #929292
    if len(x_test_) >= 14:
        ax_part.axvspan(x_test_[0], x_test_[13], facecolor ='#929292', alpha = 0.5)
    else:
        ax_part.axvspan(x_test_[0], x_test_[-1], facecolor ='#929292', alpha = 0.5)
    
    save_path = f'{path}plots_part'
    make_single_directory(save_path)
    ax_part.set_title(TITLE, fontsize=16)
    plt.savefig(f'{save_path}/{TITLE}_{cur_ID}.png', dpi=200, bbox_inches='tight')
    plt.clf()

    
    
def set_annotation(TEXT, x, y):
    # (x, y) are figure coordinates, not a data coordinates
    # (0, 0) is the bottom left, (1, 1) is the top right of the figure
    plt.gcf().text(x, y, TEXT, fontsize=20, horizontalalignment='center')
    
    

def show_histplot(df, hist=True, bins=50, kde=True, rug=False, 
                  hist_kws={'histtype':'bar',
                            'orientation': 'vertical',
                            'log': False, 
                            'stacked': False, 
                            'cumulative':False,
                            'color': 'b', 
                            'lw':None, 
                            'label':None},
                  #kde attributes
                  kde_kws={"color": "b", 
                           'shade':False},
                  vertical= False, ax=None):
        params = {'figure.figsize': (15, 5), 'font.size': 12,'axes.titlesize':'x-large', 'legend.loc': 'best', 'xtick.labelsize':'medium', 'ytick.labelsize':'medium'}
        pylab.rcParams.update(params)
        return sns.distplot(df, bins=bins, hist=hist, kde=kde, rug=rug, hist_kws=hist_kws, kde_kws=kde_kws, vertical=vertical, ax=ax)
    
    
    
def plot_train(**kwargs):
    #plot_predict.__globals__.update(kwargs)
    
    cur_ID = block_idx_ID[idx]
    
    # train X, y, prediction
    x_train = ID_to_train_block_idx[cur_ID]
    y_train = results_train.loc[x_train, 'sales']
    p_train = results_train.loc[x_train, 'pred_sales']

    # valid X, y, prediction
    x_valid = ID_to_valid_block_idx[cur_ID]
    y_valid = results_valid.loc[x_valid]['sales']
    p_valid = results_valid.loc[x_valid]['pred_sales']

    # train + valid
    x_train_valid = np.sort(np.append(x_train, x_valid))
    y_train_valid = results_train_valid.loc[x_train_valid]['sales']
    p_train_valid = results_train_valid.loc[x_train_valid]['pred_sales']

   

    # reset x_axis[0] as 0
    x_total = x_train_valid - x_train_valid[0]


    # ===== PLOT SALES 
    # ----- true, train
    fig, ax = plt.subplots()
    sns.lineplot(x=x_total, y=y_train_valid, label='true', color='black', linewidth=0.8, alpha=0.7, ax=ax)
    sns.lineplot(x=x_total, y=p_train_valid, label='train', color='#5392cd', linewidth=1.1, ax=ax)
    
       
    
    TITLE = f'Block No.{idx:04d}'
    ax.set_title(TITLE, fontsize=16)
    ax.legend(prop={'size': 14})
    # rotate xticks
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    

    # ===== TEXT for block information =====
    TEXT = f'Total length: {len(x_total)}(day)'
    # ----- table contents
    cell_text=[
        [f'{len(x_train)} day', f'{len(x_valid)} day']
        , [f'{y_train.mean():.1f}', f'{y_valid.mean():.1f}']
        , [f"{block_summary.at[idx, 'MAE_train']:.3f}", f"{block_summary.at[idx, 'MAE_valid']:.3f}"]
         ]


    # ----- colors of columns, rows
    ccolors = plt.cm.Greys(np.full(len(column_headers), 0.3))
    rcolors = plt.cm.Greys(np.full(len(row_headers), 0.3))

    # ----- make table
    table = ax2.table(colLabels=column_headers
                      , rowLabels=row_headers
                      , cellText=cell_text
                      , colWidths  = [0.1,0.1,0.1,0.1] 
                      , rowColours=rcolors
                      , colColours=ccolors
                      , cellLoc ='center'
                      , loc='center')

    table.set_fontsize(16)
    table.scale(1,2)
    ax2.axis('tight')
    ax2.axis('off')
    
    block_info = cur_ID.split('-')
    SALES_INFO = f"Total Sales: {int(block_summary.at[idx, 'sales'])}"
    INFO = f'상품유형: {block_info[0]}, 판매코드: {block_info[1]}\n\
    BS주기: {block_info[2]}, 용도구분: {block_info[3]}, 의무사용기간: {block_info[4]} \n\
    할인구분: {block_info[5]}, 할인유형: {block_info[6]}, 1+1/재렌탈: {block_info[7]}\n\
    프로모션유형: {block_info[8]}'
    
    plt.gcf().text(0.12, 0.18, TEXT + '\n'+ SALES_INFO + '\n' + '\n' + INFO, fontsize=16, horizontalalignment='left')
                  #0.15, 0.135
    
    # ===== save plot image including TEXT, TABLE
    save_path = f'{path}plots_vline'
    make_single_directory(save_path)
    plt.savefig(f'{save_path}/{TITLE}_{cur_ID}_train.png', dpi=200, bbox_inches='tight')
    plt.clf()

    
