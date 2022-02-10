from tqdm import tqdm

# markov chain monte carlo
def random_unit_vec(dim):
	vec = np.random.rand(dim) - 0.5
	length = np.linalg.norm(vec)
	return vec / length
	
print('mcmc signature:')
print('def mcmc(nb, df, num_steps, step_size, params, weights=None, filestem="weights/w", save_every=10000, p_w_save=False):')

def mcmc(nb, df, num_steps, step_size, params, weights=None, filestem="weights/w", save_every=10000, p_w_save=False):
	n = len(params)

	# init weights
	if weights is None:
		weights = np.ones(n)
	elif isinstance(weights, str):
		weights = np.load(weights )
	best_weights = None 

	# init scores, save file count, direction
	prev = 0
	score = nb.test(df, params, weights, report=False)
	print('init score:', score)
	best_score = score
	fcount = 0
	delta = random_unit_vec(n) * step_size

	for i in tqdm(range(num_steps)):
		# undo if last step lowered score
		if score < prev:
			weights -= delta
			score = prev
			delta = random_unit_vec(n) * step_size
		else:
			prev = score

		# backup current weights
		if i and i % save_every == 0:
			np.save(f'{filestem}{fcount}.npy', weights)
			if p_w_save:
				print(f'score at step {i}: {score}')
			fcount += 1

		# step and re-score
		weights += delta
		score = nb.test(df, params, weights, report=False)

		# update best
		if score > best_score:
			best_score = score
			best_weights = np.copy(weights)

	print(best_score)
	print(best_weights)
	np.save(f'{filestem}_best.npy', best_weights)
	return best_score, best_weights
