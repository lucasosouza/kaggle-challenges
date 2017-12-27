# run.py
from load import gen_data, gen_batch
from preprocess import prep_train_data, prep_test_data
from nn import neural_net  
from export import get_added_products
from numpy import arange

# def run():

#     ## first load train and test data
#     X, y, labels =  gen_data()
#     X_test = load_test_data()

#     ## then preprocess training and testing dataset
#     X,y,X_test, labels, last_row = prep_data(X, y, labels, X_test)

#     ## then train dataset 
#     preds = train_and_predict(X, y, X_test)

#     ## finally export 
#     get_added_products(preds, X_test, labels, last_row)

# def predict():

#     ## first load train and test data
#     X, y, labels =  gen_data()
#     X_test = load_test_data()

#     ## then preprocess training and testing dataset
#     X,y,X_test, labels, last_row = prep_data(X, y, labels, X_test)

#     ## then predict dataset
#     preds = only_predict(X_test)

#     ## finally export 
#     get_added_products(preds, X_test, labels, last_row)

def run_in_batches():

    ## load test data
    X_train, y_train, labels = gen_data()
    X_test = gen_data(test=True, labels=labels)

    ## prep test data
    X_test, features, current_products = prep_test_data(X_train, y_train, X_test) 

    ## setup training
    sess = None
    step = .02
    ## divide training in batches 
    for batch in arange(0, 1, step):
        print("{:.2f}-{:.2f}".format(batch, batch+step))
        ## load partial train data
        X_batch, y_batch =  gen_batch(X_train, y_train, batch, step, labels) 
        ## prep data
        X_batch, y_batch = prep_train_data(X_batch, y_batch, features) 
        ## train
        sess = neural_net(sess, X_batch, y_batch) # ok

    ## once finished, predict
    preds = neural_net(sess, X_test, pred=True) # ok

    ## finally export 
    get_added_products(preds, X_test, labels, current_products)


def predict_from_trained():

    ## load test data
    X_train, y_train, labels = gen_data()
    X_test = gen_data(test=True, labels=labels)

    ## prep test data
    X_test, features, current_products = prep_test_data(X_train, y_train, X_test) 

    # ## setup training
    sess = None
    # step = .02
    # ## divide training in batches 
    # for batch in arange(0, 1, step):
    #     print("{:.2f}-{:.2f}".format(batch, batch+step))
    #     ## load partial train data
    #     X_batch, y_batch =  gen_batch(X_train, y_train, batch, step, labels) 
    #     ## prep data
    #     X_batch, y_batch = prep_train_data(X_batch, y_batch, features) 
    #     ## train
    #     sess = neural_net(sess, X_batch, y_batch) # ok


    ## once finished, predict
    preds = neural_net(sess, X_test, pred=True) # ok

    ## finally export 
    get_added_products(preds, X_test, labels, current_products)


if __name__=='__main__':
    predict_from_trained()
    # gen_partial_data(.9, .05, [])