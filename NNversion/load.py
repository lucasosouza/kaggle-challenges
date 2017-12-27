# import useful stuff
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from time import time

# avoid undefined metric warning when calculating precision with 0 labels defined as 1
import warnings
warnings.filterwarnings('ignore')

# functions

def gen_data(test=False, labels=None):
    if not test:
        # load train data
        df = pd.read_csv('train_ver2.csv')
        # identify labels
        labels = []
        for col in df.columns:
            if col[:4] == 'ind_' and col[-4:] == 'ult1':
                labels.append(col)
        # set NAs in labels to 0
        y = df[labels].fillna(value=0) 
    else: 
        # load test data
        df = pd.read_csv('test_ver2.csv')
    
    # create X and y, and delete dataframe
    X = df[df.columns.difference(labels)]
    del df

    # if train, also return y and labels
    if not test:
        return X,y, labels
    else:
        return X


def gen_batch(X, y, batch, step, labels):
    
    #separate per batch
    lower_lim = X['ncodpers'].quantile(batch)
    upper_lim = X['ncodpers'].quantile(batch + step)
    X = X[(X['ncodpers'] > lower_lim) & (X['ncodpers'] <= upper_lim)]
    y = y.loc[X.index]
    print('threshold: ', X.shape, y.shape)

    return X,y


def transform(df, fillna=True): 
    """ This version includes variables considered relevant"""
    
    ### variables to be removed ###
    # remove cod_prov only, since it is redundant with nomprov
    # removed fecha_alta - redundant with antiguedad
    for col in ['cod_prov', 'fecha_alta', 'ult_fec_cli_1t', 'pais_residencia']:
        del df[col]    

    # use less memory

    ### numerical_vars ###
    # convert numerical vars to int
    numerical_vars = ['age', 'antiguedad', 'renta']
    df[numerical_vars] = df[numerical_vars].convert_objects(convert_numeric=True)
    
    # change less or equal than 0 to nan
    for var in numerical_vars:
        df.ix[df[var] < 0, var] = np.nan

    ### boolean and categorical vars ###
    # convert S/N to boolean and remaining to number
    boolean_vars = ['indfall', 'ind_actividad_cliente', 'ind_nuevo', 'indresi', 'indext', 
                    'tipodom', 'conyuemp', 'ind_actividad_cliente']
    for var in ['indfall', 'indresi', 'indext', 'conyuemp']:
        df[var] = df[var] == 'S'
    df[boolean_vars] = df[boolean_vars].convert_objects(convert_numeric=True)
        
    # one hot encode categorical vars
    # 150 canais, 103 paises, 52 provincias
    categorical_vars = ['segmento', 'sexo', 'tiprel_1mes', 'canal_entrada', 
                        'ind_empleado', 'indrel_1mes', 'nomprov'] #removed nomprov - faster
    df = pd.get_dummies(df, prefix=None, prefix_sep='_', dummy_na=False, 
                       columns=categorical_vars, sparse=False, drop_first=False)    

    
    ### handling null values ###
    if fillna:
        df = df.fillna(value=0)
    else:
        df = df.dropna()
        
    ### end ### 
            
    return df

def get_delta(X, y=None):

    # sort by ncodpers and fecha_dato
    X = X.sort_values(['ncodpers', 'fecha_dato'])

    # create differences
    X_diff = X.drop(['fecha_dato'], axis=1).diff()

    # set index and ncodpers as column
    X_diff['index'] = X_diff.index
    X_diff['ncodpers'] = X['ncodpers']

    # remove first column
    first = X_diff.groupby('ncodpers').first().reset_index()['index']
    X = X_diff[-X_diff.index.isin(first)]

    if type(y)==pd.core.frame.DataFrame or type(y)==np.ndarray:
        # sort by ncodpers and fecha_dato
        y = y.loc[X.index]
        # create differences
        y_diff = y.diff()
        # remove negative y
        y_diff[y_diff < 0] = 0

    return X,y

def scale(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


# def load_data():
#     ## get the last date from training data
#     X, y, labels =  gen_data()
#     X_previous = X[X['fecha_dato']=='2016-05-28']
#     print('previous: ', X_previous.shape)

#     ## get current products
#     current_products = y[X['fecha_dato']=='2016-05-28']
#     current_products.index=X[X['fecha_dato']=='2016-05-28']['ncodpers']
#     print('current products: ', current_products.shape)

#     ## garbage collect
#     del X,y

#     ## get the next date from test data and concat
#     X_next, _, _ = gen_data(test=True)
#     X = pd.concat([X_previous, X_next])
#     print('next: ', X_next.shape)

#     ## preprocess
#     X = transform(X)
#     features = X.columns
#     print('transform: ', X.shape)

#     ## get difference
#     X, _ = get_delta(X)
#     print('delta: ', X.shape)

#     ## save columns and index
#     features, ids = X.columns, X['ncodpers']

#     ## scale
#     X = scale(X)
#     print('scale: ', X.shape)

#     return pd.DataFrame(X, columns=features, index=ids), features, labels, current_products



