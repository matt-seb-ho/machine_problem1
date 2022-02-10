from tqdm import tqdm
# age, gender, height_cm, weight_kg, body fat_%, diastolic, systolic, grip_force, sit_and_bend_forward_cm, sit_up_count, broad_jump_cm

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
def experiment2(params):
	top_score = 0
	top_params = []
	for i in range(1, 2 ** len(params)):
		temp = num_to_params(i, params)
		score = nb.test(test_df, temp, report=False)
		if score > top_score:
			top_score = score
			top_params = temp
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
