import numpy as np
import pandas as pd
from time import perf_counter
import sys

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
					self.discrete_llh[label][c] = subset[c].value_counts() / len(subset)
			
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

header = [
	'age', 'gender', 'height_cm', 'weight_kg', 
	'body_fat_pct', 'diastolic', 'systolic', 'grip_force', 
	'sit_and_bend_forward_cm', 'sit_up_count', 'broad_jump_cm', 'label'
]

all_params = [
	'age', 'gender', 'height_cm', 'weight_kg', 
	'body_fat_pct', 'diastolic', 'systolic', 'grip_force', 
	'sit_and_bend_forward_cm', 'sit_up_count', 'broad_jump_cm'
]

my_params = ['gender', 'weight_kg', 'diastolic', 'grip_force', 'sit_and_bend_forward_cm', 'sit_up_count']

if __name__ == "__main__":
	# train_fname = 'train.txt'
	# test_fname = 'test.txt'
	# print('header:', header)

	if len(sys.argv) < 3:
		sys.exit('please supply 2 command line arguments: [train filename] [test filename]')

	train_fname = sys.argv[1]
	test_fname = sys.argv[2]
	train_df = pd.read_csv(train_fname, names=header)
	test_df = pd.read_csv(test_fname, names=header)

	nb = NBClassifier()
	nb.train(train_df, print_time=False)

	# personal testing vs gradescope submission
	if len(sys.argv) == 3:
		nb.test(test_df, my_params, print_pred=True, report=False)
	else:
		nb.test(test_df, my_params, print_pred=False, report=True)

	# top10 = test_df[:10]
	# nb.test(top10, params, print_pred=True, report=False)
