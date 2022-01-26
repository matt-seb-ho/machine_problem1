import numpy as np
import pandas as pd
from scipy.stats import norm
from time import perf_counter

# constant for normpdf
root2pi = np.sqrt(2 * np.pi)

class NBClassifier:
	def __init__(self):
		# gaussian distribution parameters for continuous variables, indexed on label
		self.mu = [None, None]
		self.sigma = [None, None]

		# gender likelihoods
		# gender_llh[l][g] = P(gender=g | label=l) = #(g, l) / #(l)
		self.gender_llh = [{}, {}]

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
			self.gender_llh[label]['F'] = len(subset[subset['gender'] == 'F']) / len(subset)
			self.gender_llh[label]['M'] = len(subset[subset['gender'] == 'M']) / len(subset)
			self.label_counts[label] = len(subset)
			
		self.train_size = len(df)
		stop = perf_counter()
		print(f"trained classifier in {stop - start:0.3f} seconds")

	def normpdf(self, col, mean, std):
		return (
			np.exp(
				-0.5 * (((col - mean) / std) ** 2)
			) / (std * root2pi)
		)
	
	def test(self, df, params, print_pred=False):
		start = perf_counter()
		df['p0'] = 0
		df['p1'] = 0
		for param in params:
			if param == 'gender':
				# df['p0'] += np.log2(df['gender'].apply(
				# 	lambda x : self.gender_counts[0][x] / self.label_counts[0]
				# ))
				# df['p1'] += np.log2(df['gender'].apply(
				# 	lambda x : self.gender_counts[1][x] / self.label_counts[1]
				# ))
				
				df['p0'] = np.log2(df['gender'].map(self.gender_llh[0]))
				df['p1'] = np.log2(df['gender'].map(self.gender_llh[1]))


			else:
				# gauss0 = norm(self.mu[0][param], self.sigma[0][param])
				# gauss1 = norm(self.mu[1][param], self.sigma[1][param])
				# df['p0'] += np.log2(df[param].apply(lambda x : gauss0.pdf(x)))
				# df['p1'] += np.log2(df[param].apply(lambda x : gauss1.pdf(x)))

				df['p0'] += np.log2(self.normpdf(df[param], self.mu[0][param], self.sigma[0][param]))
				df['p1'] += np.log2(self.normpdf(df[param], self.mu[1][param], self.sigma[1][param]))

		correct = 0
		# for idx, row in df.iterrows():
		# 	guess = 0 if row['p0'] > row['p1'] else 1
		# 	if guess == row['label']:
		# 		correct += 1
		# 		if print_pred:
		# 			print(guess)
		
		pred = df['p1'] > df['p0'] 
		if print_pred:
			print(pred.astype('i1').to_string(index=False))
		correct = (pred == df['label']).sum()

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
print('header:', header)
train_df = pd.read_csv(train_fname, names=header)
test_df = pd.read_csv(test_fname, names=header)

nb = NBClassifier()
nb.train(train_df)
nb.test(test_df, params)

nb.test(test_df, ['gender'])

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
			top_params = temp
	print('--------\nThe Best:')
	print(top_score)
	print(top_params)

# experiment1()
