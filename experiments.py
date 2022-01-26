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

def num_to_params(n):
	res = []
	idx = 0
	while n > 0:
		if n & 1:
			res.append(params[idx])
		n >>= 1;
		idx += 1
	return res

# exhaustive search: test all possible combination of parameters
def experiment2():
	top_score = 0
	top_params = []
	for i in range(1, 2048):
		temp = num_to_params(i)
		score = nb.test(test_df, temp, report)
		if score > top_score:
			top_score = score
			top_params = temp
	print('--------\nThe Best:')
	print(top_score)
	print(top_params)

# experiment2()
# exp2 results: ['gender', 'weight_kg', 'diastolic', 'grip_force', 'sit_and_bend_forward_cm', 'sit_up_count']

	
# markov chain monte carlo
def random_unit_vec(dim):
	vec = np.random.rand(dim) - 0.5
	length = np.linalg.norm(vec)
	return vec / length
	
def mcmc(nb, test, params, num_steps, step_size, starter_weights=None, filestem="weights/w", save_every=10000):
	n = len(params)

	# init weights
	if starter_weights is None:
		starter_weights = np.ones(n)
	elif isinstance(starter_weights, str):
		starter_weights = np.load(starter_weights)
	weights = starter_weights
	best_weights = starter_weights

	# init scores, save file count, direction
	prev = 0
	score = nb.test(test, params, weights, report=False)
	best_score = score
	fcount = 0
	delta = random_unit_vec(n) * step_size

	for i in range(num_steps):
		# backup current weights
		if i and i % save_every == 0:
			np.save(f'{filestem}{fcount}.npy', weights)
			print(f'score at step {i}: {score}')
			fcount += 1

		# undo if last step lowered score
		if score < prev:
			weights -= delta
			delta = random_unit_vec(n) * step_size

		# step and re-score
		prev = score
		weights += delta
		score = nb.test(test, params, weights, report=False)

		# update best
		if score > best_score:
			best_score = score
			best_weights = weights

	print(best_score)
	print(best_weights)
	np.save(f'{filestem}_best.npy', best_weights)
	return best_score, best_weights
