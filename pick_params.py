import numpy as np
from tqdm import tqdm

# base params:  age, gender, height_cm, weight_kg, body fat_%, diastolic, systolic, grip_force, sit_and_bend_forward_cm, sit_up_count, broad_jump_cm
# custom params: bp_class, bmi, bmi_class

def num_to_params(n, params):
	res = []
	idx = 0
	while n > 0:
		if n & 1:
			res.append(params[idx])
		n >>= 1;
		idx += 1
	return res

# exhaustive search: test all possible combination of parameters
def check_all(params, test_df, save_f):
	top_score = 0
	top_params = []
	num_setups = 2 ** len(params)
	scores = np.zeros(num_setups)
	for i in tqdm(range(1, num_setups)):
		temp = num_to_params(i, params)
		score = nb.test(test_df, temp, report=False)
		scores[i] = score
		if score > top_score:
			top_score = score
			top_params = temp

	np.save(save_f + ".npy", scores)
	with open(save_f + "_key.txt", "w") as f:
		f.write(f'{save_f}.npy encoding is based on this param list: {str(params)}')
	
	print('--------\nThe Best:')
	print(top_score)
	print(top_params)

# experiment2()
# exp2 results: ['gender', 'weight_kg', 'diastolic', 'grip_force', 'sit_and_bend_forward_cm', 'sit_up_count']
remaining = ['age', 'height_cm', 'body_fat_pct', 'systolic', 'broad_jump_cm']

def experiment3(tf, params, base_p, base_w):
	top_score = 0
	top_params = []

	for i in range(2 **len(params)):
		temp = num_to_params(i, params)
		addlen = len(temp)
		temp_p = base_p.copy()
		temp_p.extend(temp)

		weights = np.copy(base_w)
		weights = np.pad(weights, pad_width=(0, addlen), constant_values=(1))
		print(temp_p, '\n', weights)
		score = nb.test(tf, temp_p, weights)
		if score > top_score:
			top_score = score
			top_params = temp
	 
	print('--------\nThe Best:')
	print(top_score)
	print(top_params)
