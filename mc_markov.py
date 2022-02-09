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
	best_weights = None 

	# init scores, save file count, direction
	prev = 0
	score = nb.test(test, params, weights, report=False)
	print('init score:', score)
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
		else:
			prev = score

		# step and re-score
		weights += delta
		score = nb.test(test, params, weights, report=False)

		# update best
		if score > best_score:
			best_score = score
			best_weights = np.copy(weights)

	print(best_score)
	print(best_weights)
	np.save(f'{filestem}_best.npy', best_weights)
	return best_score, best_weights
