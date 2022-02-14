import numpy as np
import pandas as pd
from time import perf_counter
import sys
from collections import defaultdict

# constant for normpdf
root2pi = np.sqrt(2 * np.pi)

class NBClassifier:
    def __init__(self):
        # gaussian distribution parameters for continuous variables, indexed on label
        self.mu = [None, None]
        self.sigma = [None, None]

        # discrete variable likelihoods, indexed on label
        self.discrete_llh = [{}, {}]

    def train(self, df, print_time=False):
        # for each label, gather continuous distribution params and discrete variable counts
        start = perf_counter()
        for label in [0, 1]:
            subset = df[df['label'] == label]
            self.mu[label] = subset.mean(numeric_only=True)
            self.sigma[label] = subset.std(numeric_only=True)

            # MLE for discrete variables
            for c in df.columns:
                if df[c].dtypes == np.object:
                    self.discrete_llh[label][c] = defaultdict(
                        lambda : 1,
                        (subset[c].value_counts() / len(subset)).to_dict()
                    )
            
        stop = perf_counter()
        if print_time:
            print(f"trained classifier in {stop - start:0.3f} seconds")

    def normpdf(self, col, mean, std):
        return (
            np.exp(
                -0.5 * (((col - mean) / std) ** 2)
            ) / (std * root2pi)
        )
    
    def test(self, df, params, weights=None, print_pred=False, report=True):
        start = perf_counter()
        df['p0'] = 0
        df['p1'] = 0
        if weights is None:
            weights = [1] * len(params)

        for param, w in zip(params, weights):
            if df[param].dtypes == np.object:
                # use MLE for discrete variables 
                df['p0'] += w * np.log2(df[param].map(
                    lambda x : self.discrete_llh[0][param][x]
                ))
                df['p1'] += w * np.log2(df[param].map(
                    lambda x : self.discrete_llh[1][param][x]
                ))

            else:
                # use Gaussian Distribution for continuous variables
                df['p0'] += w * np.log2(self.normpdf(df[param], self.mu[0][param], self.sigma[0][param]))
                df['p1'] += w * np.log2(self.normpdf(df[param], self.mu[1][param], self.sigma[1][param]))

        correct = 0
        pred = df['p1'] > df['p0'] 
        if print_pred:
            print(pred.astype('i1').to_string(index=False))
        correct = (pred == df['label']).sum()

        accuracy = correct / len(df)
        if report:
            print(f"accuracy: {accuracy}")
            stop = perf_counter()
            print(f"ran inference in {stop - start:0.3f} seconds")
        return accuracy
