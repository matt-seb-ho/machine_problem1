from tqdm import tqdm
import numpy as np

def random_unit_vec(dim):
    vec = np.random.rand(dim) - 0.5
    length = np.linalg.norm(vec)
    return vec / length

print('def mcmc(nb, df, ps, ws, steps, step_size, save_f="weights/mc2_", '
      'save_every=1e10, decay_steps=50, decay_rate=.9):')

# Monte Carlo Markov Chain: random walk for "learning" weights
def mcmc(nb, df, ps, ws, steps, step_size, save_f="weights/mc2_", 
         save_every=1e10, decay_steps=50, decay_rate=.9):
    # initialization step
    dims = len(ps)
    sc = nb.test(df, ps, ws, report=False)
    best_sc = sc 
    best_ws = np.copy(ws)
    print('initial score:', sc)

    step = random_unit_vec(dims) * step_size
    next_ws = ws + step
    next_sc = nb.test(df, ps, next_ws, report=False)

    stuck_steps = 0 
    stuck_threshold = 250
    decay_every = steps // decay_steps

    for i in tqdm(range(steps)):
        # save periodically
        if i and i % save_every == 0:
            np.save(save_f + str(i) + ".npy", ws)

        # decay step size
        if i and i % decay_every:
            step_size *= decay_rate

        if sc <= best_sc:
            stuck_steps += 1

        if stuck_steps > stuck_threshold:
            # if stuck for too long, take random step
            ws += 2 * random_unit_vec(dims) * step_size
            sc = nb.test(df, ps, next_ws, report=False)
            stuck_steps = 0

        elif next_sc < sc:
            # if next score is worse, pick a different next_ws
            step = random_unit_vec(dims) * step_size
        else:
            # update score and weights
            sc = next_sc
            ws = next_ws
            # update global best
            if sc > best_sc:
                best_sc = sc
                best_ws = np.copy(ws)
                stuck_steps = 0

        # take the next step
        next_ws = ws + step
        next_sc = nb.test(df, ps, next_ws, report=False)
        #  print(next_sc, next_ws)
    print(sc, ws)
    np.save(f'{save_f}fin.npy', best_ws)
    return best_sc, best_ws
