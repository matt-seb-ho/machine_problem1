import numpy as np
import pandas as pd

# data is numpy array of shape Nx2 where 
def impurity(data):
    n = len(data)
    p1 = np.count_nonzero(data[:, 1]) / n
    p0 = 1 - p1
    return 1 - (p0 ** 2 + p1 ** 2)
    

# sort data routine
# sd = data[np.argsort(data[:, 0])]

def prep_disc_data(df, feature):
    f_only = (df[[feature, 'label']]).to_numpy()
    sd = f_only[np.argsort(f_only[:, 0])]
    return sd

# assume sorted data
def find_bounds(data, max_depth=2):
    n = len(data)

    best_split = None
    min_impure = None
    best_idx = None
    # check every split
    for i in range(n - 1):
        # split data
        less_split = data[:i + 1]
        more_split = data[i + 1:]

        # eval split
        w_impure = ((len(less_split) / n) * impurity(less_split)
                    + ((len(more_split) / n) * impurity(more_split)))
        if min_impure is None or w_impure < min_impure:
            min_impure = w_impure
            best_split = 0.5 * (less_split[-1, 0]  + more_split[0, 0])
            best_idx = i

    res = [best_split]
    if max_depth != 1:
        less_split = data[:best_idx + 1]
        more_split = data[best_idx + 1:]
        if impurity(less_split) != 0:
            l_splits = find_bounds(less_split, max_depth=max_depth-1)
            res = l_splits + res
        if impurity(more_split) != 0:
            m_splits = find_bounds(more_split, max_depth=max_depth-1)
            res = res + m_splits
    return res


def discretize(train_df, test_df, feature, split_depth=2):
    sd = prep_disc_data(train_df, feature)
    bounds = find_bounds(sd, max_depth=split_depth)
    def cat_fn(row):
        for i, bound in enumerate(bounds):
            if row[feature] < bound:
                return str(i)
        return str(len(bounds))
    train_df[feature+"_d"] = train_df.apply(cat_fn, axis=1)
    test_df[feature+"_d"] = test_df.apply(cat_fn, axis=1)
