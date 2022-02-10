from tqdm import tqdm
import numpy as np

def code_to_diff(n, num_params, scale):
    res = np.zeros(num_params)
    idx = 0
    while n > 0:
        if n & 1:
            res[idx] = 1 if n & 2 else -1
        elif n & 2:
            return None 
        n >>= 2
        idx += 1
    res *= scale
    return res


def make_step_mat(num_params, save_file=None):
    lst = []
    for n in range(4 ** num_params):
        diff = code_to_diff(n, num_params, 1)
        if diff is not None:
            lst.append(diff)
    mat = np.stack(lst)
    print('^-^)b done!')
    if save_file is not None:
        np.save(save_file, mat)
    return mat

sm6d = np.load("grid_step_6d.npy")
print('grid sig: def grid(nb, df, params, weights, scale, disp=True):')
def grid(nb, df, params, weights, scale, sm=sm6d, disp=True):
    init_score = nb.test(df, params, weights, report=False)
    top_score = init_score
    top_weights = np.copy(weights)
   
    lst = sm6d * scale
    if disp:
        lst = tqdm(lst)

    for step in lst:
        score = nb.test(df, params, weights + step, report=False)
        if score > top_score:
            top_score = score
            top_weights = np.copy(weights + step)
    if disp:
        print(f'score after: {top_score}')
        if init_score == top_score:
            print('no improvement ;-;')
    return top_score, top_weights, top_score > init_score

print('igs sig: def iterated_gs(nb, df, params, weights, scale, max_iter=100, save_file=None, disp=True):')

def iterated_gs(nb, df, params, weights, scale=.1, max_iter=100, save_file=None, disp=True):
    score = nb.test(df, params, weights, report=False)
    improving = True
    iters = 0
    desperate = [0.05, 0.02, 0.01, 0.005, 0.001]
    while improving and iters < max_iter:
        score, weights, improving = grid(nb, df, params, weights, scale, disp)
        iters += 1
        if not improving:
            for pls in desperate:
                score, weights, improving = grid(nb, df, params, weights, pls, disp)
                iters += 1
                if improving:
                    break
    print(score)
    if save_file is not None:
        np.save(save_file, weights)

def random_unit_vec(dim):
	vec = np.random.rand(dim) - 0.5
	length = np.linalg.norm(vec)
	return vec / length

print('stochastic sig: def hybrid_search(nb, df, ps, weights, scale=.1, max_steps=200, save_file="weights/dummy.npy", bar=True):')
milestones = { round(0.01 * pct, 2) : False for pct in range(82, 89) }
decayed_steps = [0.05, 0.01, 0.001]
def hybrid_search(
        nb, df, ps, weights, 
        scale=.1, max_steps=200, save_every=1e10,
        save_file="weights/dummy", bar=True):
    score = nb.test(df, ps, weights, report=False)
    improving = True
    milestone = .82
    looper = tqdm(range(max_steps)) if bar else range(max_steps)
    for idx in looper:
        # if we haven't reached this milestone yet
        if score > milestone:
            np.save(save_file + f'_m{milestone}.npy', weights)
            milestones[milestone] = True
            milestone += 0.01
            milestones[milestone] = False

        if idx and idx % save_every == 0:
            np.save(save_file + str(idx) + ".npy", weights)
        score, weights, improving = grid(nb, df, ps, weights, scale, disp=False)
        if not improving:
            for decayed in decayed_steps:
                score, weights, improving = grid(nb, df, ps, weights, scale, disp=False)
                if improving:
                    break
        if not improving:
            weights += 2 * scale * (random_unit_vec(len(ps)))
    print(f'final score: {score}')
    np.save(save_file + ".npy", weights)
