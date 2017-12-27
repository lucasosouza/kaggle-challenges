# export.py
# load training data

import pandas as pd
import numpy as np

def get_added_products(preds, X_test, labels, last_row):

    ## consolidate products in last row
    def consolidate_products(arr):
        products_list = []
        for i, item in enumerate(arr):
            if item>0:
                products_list.append(i)
        return products_list
    last_row['products'] = last_row.apply(consolidate_products, axis=1)

    # construct results data frame
    results = pd.DataFrame(preds.indices, index=X_test.index)

    # consolidate results
    def concat_products(row):
        return tuple([int(i) for i in row])
    results['new_products'] = results.apply(concat_products, axis=1, reduce=True, raw=True)
    results['new_products'].head()

    # concatenate two dataframes
    diff = pd.DataFrame(index=X_test.index)
    diff['products'] = last_row['products']
    diff['new_products'] = results['new_products']

    # get the difference
    def diff_products(row):
        return [i for i in row[1] if i not in row[0]]
    diff['added_products'] = diff.apply(diff_products, axis=1)

    # change i for string and join
    def convert_products(row):
        mod_row = map(lambda x: labels[x], row)
        return " ".join(mod_row)
        
    diff['added_products'] = diff['added_products'].apply(convert_products)

    # export
    del diff['products']
    del diff['new_products']
    diff.to_csv('round10c.csv', header=True, index=True)

    print("Exported successfully: ", diff.shape)
