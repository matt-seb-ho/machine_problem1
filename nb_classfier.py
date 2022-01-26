import numpy as np
import pandas as pd
from scipy.stats import norm
from time import perf_counter

class NBClassifier:
	def __init__(self):
		# gaussian distribution parameters for continuous variables, indexed on label
		self.mu = [None, None]
		self.sigma = [None, None]

		# gender_counts[l][g] = # of training points with label l and gender g
		self.gender_counts = [{}, {}]

		# label_counts 
		self.label_counts = [None, None]
		self.train_size = None

	def train(self, df):
		# for each label, gather continuous distribution params and discrete variable counts
		start = perf_counter()
		for label in [0, 1]:
			subset = df[df['label'] == label]
			self.mu[label] = subset.mean(numeric_only=True)
			self.sigma[label] = subset.std(numeric_only=True)
			self.gender_counts[label]['F'] = len(subset[subset['gender'] == 'F'])
			self.gender_counts[label]['M'] = len(subset) - self.gender_counts[label]['F']
			self.label_counts[label] = len(subset)
		self.train_size = len(df)
		stop = perf_counter()
		print(f"trained classifier in {stop - start:0.3f} seconds")
	
	def test(self, df, params, print_pred=False):
		start = perf_counter()
		df['p0'] = 0
		df['p1'] = 0
		for param in params:
			if param == 'gender':
				df['p0'] += np.log2(df['gender'].apply(
					lambda x : self.gender_counts[0][x] / self.label_counts[0]
				))
				df['p1'] += np.log2(df['gender'].apply(
					lambda x : self.gender_counts[1][x] / self.label_counts[1]
				))
			else:
				gauss0 = norm(self.mu[0][param], self.sigma[0][param])
				gauss1 = norm(self.mu[1][param], self.sigma[1][param])
				df['p0'] += np.log2(df[param].apply(lambda x : gauss0.pdf(x)))
				df['p1'] += np.log2(df[param].apply(lambda x : gauss1.pdf(x)))
		correct = 0
		for idx, row in df.iterrows():
			guess = 0 if row['p0'] > row['p1'] else 1
			if guess == row['label']:
				correct += 1
				if print_pred:
					print(guess)
		accuracy = correct / len(df)
		print(accuracy)
		stop = perf_counter()
		print(f"ran inference in {stop - start:0.3f} seconds")
		return accuracy


header = [
	'age', 'gender', 'height_cm', 'weight_kg', 
	'body_fat_pct', 'diastolic', 'systolic', 'grip_force', 
	'sit_and_bend_forward_cm', 'sit_up_count', 'broad_jump_cm', 'label'
]

params = [
	'age', 'gender', 'height_cm', 'weight_kg', 
	'body_fat_pct', 'diastolic', 'systolic', 'grip_force', 
	'sit_and_bend_forward_cm', 'sit_up_count', 'broad_jump_cm'
]


train_fname = 'train.txt'
test_fname = 'test.txt'
train_df = pd.read_csv(train_fname, names=header)
test_df = pd.read_csv(test_fname, names=header)

nb = NBClassifier()
nb.train(train_df)
nb.test(test_df, params)

# experiment 1: try dropping single parameters
def experiment1():
	top_score = 0
	top_params = []
	for param in params:
		temp = params.copy()
		temp.remove(param)
		score = nb.test(test_df, temp)
		if score > top_score:
			top_score = score
			top_params = params
	print('--------\nThe Best:')
	print(top_score)
	print(top_params)

# experiment1()
