import numpy as np
import random

# greedy search strategy: 
# pass: adjust one weight at a time until there is no improvement 
# shuffle weight order each pass

params = ['gender', 'weight_kg', 'diastolic', 'grip_force', 'sit_and_bend_forward_cm', 'sit_up_count']
score, p_score, weights, p_weights = None, None, None, None
weights_prefix = 'weights/w'
save_freq = 10000

# I want to repeatedly:
# 0. save score and weights under previous
# 1. nudge single weight
# 2. store new score

def step(nb, test, param_idx, diff):
    global score, p_score, weights, p_weights
    p_score, p_weights = score, np.copy(weights)
    weights[param_idx] += diff
    score = nb.test(test, params, weights, report=False)
    # print(score)

# pre-condition: score and prev_score are initialized
def single_pass(nb, test, order, step_size):
    global score, p_score, weights, p_weights
    # for each parameter, in the specified order
    for param_idx in order:
        # first try increasing weight
        step(nb, test, param_idx, step_size)
        improved = False
        while score > p_score:
            improved = True
            step(nb, test, param_idx, step_size)
        # if no improvement, try the other direction
        if not improved:
            score = p_score
            weights[param_idx] -= step_size
            # s == p_s && w == w_s
            step(nb, test, param_idx, -step_size)
            while score > p_score:
                improved = True
                step(nb, test, param_idx, -step_size)
            if not improved:
                score = p_score
                weights[param_idx] += step_size
        
    
def greedy_search(nb, test, epochs=1, step_size=0.01):
    global score, p_score, weights, p_weights
    for i in range(epochs):
        order = list(range(len(params))) 
        random.shuffle(order)
        single_pass(nb, test, order, step_size)
