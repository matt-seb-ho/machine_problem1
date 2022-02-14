import numpy as np
import pandas as pd
from nb_classifier import *
from custom_features import *
from discretize import discretize

header = [
	'age', 'gender', 'height_cm', 'weight_kg', 
	'body_fat_pct', 'diastolic', 'systolic', 'grip_force', 
	'sit_and_bend_forward_cm', 'sit_up_count', 'broad_jump_cm', 'label'
]
all_params = [
	'age', 'gender', 'height_cm', 'weight_kg', 
	'body_fat_pct', 'diastolic', 'systolic', 'grip_force', 
	'sit_and_bend_forward_cm', 'sit_up_count', 'broad_jump_cm'
]
# my_params = ['gender', 'weight_kg', 'diastolic', 'grip_force', 'sit_and_bend_forward_cm', 'sit_up_count']
my_params = ['gender', 'weight_kg', 'grip_force', 'sit_and_bend_forward_cm', 'sit_up_count', 'bp_class']
wstart = np.load("weights/mc2p2_fin.npy")

train_fname = 'train.txt'
test_fname = 'test.txt'

train_df = pd.read_csv(train_fname, names=header)
test_df = pd.read_csv(test_fname, names=header)
add_custom_features([train_df, test_df], all_params)


myp = []
make_discrete = True 

if make_discrete:
    c_to_d = [
        'age', 'height_cm', 'weight_kg', 
        'body_fat_pct', 'diastolic', 'systolic', 'grip_force', 
        'sit_and_bend_forward_cm', 'sit_up_count', 'broad_jump_cm'
    ]

    for feature in c_to_d:
        discretize(train_df, test_df, feature, split_depth=3)

    myp = [x + '_d' for x in c_to_d]
    myp.append('gender')

nb = NBClassifier()
nb.train(train_df, print_time=False)

print("Loaded train and test sets into dfs; Initialized nb trained on train_df; Defined header, all_params, and my_params")
