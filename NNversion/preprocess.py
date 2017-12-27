# run.py
from load import *

def prep_train_data(X, y, features):

    ## preprocess training data
    X = transform(X)
    print('transform: ', X.shape)

    ## get difference
    X, y = get_delta(X, y)
    print('delta: ', X.shape, y.shape)

    ## look for features not available in train dataset
    missing_features = [col for col in features if col not in X.columns]
    # pad with 0
    for f in missing_features:
        X[f] = 0
    # filter X to same features in X_train
    X = X[features] 
    print('paired features: ', X.shape)

    ## save columns and index
    features_train, ids_train = X.columns, X['ncodpers']

    ## scale
    X = scale(X)
    print('scale: ', X.shape)

    ## convert back to pandas
    X = pd.DataFrame(X, columns=features_train, index=ids_train)

    return X,y


def prep_test_data(X, y, X_test):

    ## get the last date from training data
    X_previous = X[X['fecha_dato']=='2016-05-28']
    print('previous: ', X_previous.shape)

    ## get current products
    current_products = y[X['fecha_dato']=='2016-05-28']
    current_products.index=X[X['fecha_dato']=='2016-05-28']['ncodpers']
    print('current products: ', current_products.shape)

    ## garbage collection
    del X, y

    ## get the next date from test data and concat
    X = pd.concat([X_previous, X_test])
    print('next: ', X.shape)

    ## preprocess
    X = transform(X)
    features = X.columns
    print('transform: ', X.shape)

    ## get difference
    X, _ = get_delta(X)
    print('delta: ', X.shape)

    ## save columns and index
    features, ids = X.columns, X['ncodpers']

    ## scale
    X = scale(X)
    print('scale: ', X.shape)

    ## convert back to df
    X = pd.DataFrame(X, columns=features, index=ids)
    ## look for features not available in train dataset
    missing_features = [col for col in features if col not in X.columns]
    # pad with 0
    for f in missing_features:
        X[f] = 0
    # filter X to same features in X_train
    X = X[features] 
    print('paired features: ', X.shape)

    return X, features, current_products

