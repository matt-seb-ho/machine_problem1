import numpy as np
import pandas as pd
import nb_classifier as nb_mod

header = [
	'age', 'gender', 'height_cm', 'weight_kg', 
	'body_fat_pct', 'diastolic', 'systolic', 'grip_force', 
	'sit_and_bend_forward_cm', 'sit_up_count', 'broad_jump_cm', 'label'
]

my_params = ['gender', 'weight_kg', 'diastolic', 'grip_force', 'sit_and_bend_forward_cm', 'sit_up_count']

train_fname = 'train.txt'
test_fname = 'test.txt'
print('header:', header)
train_df = pd.read_csv(train_fname, names=header)
test_df = pd.read_csv(test_fname, names=header)

nb = nb_mod.NBClassifier()
nb.train(train_df, print_time=False)
