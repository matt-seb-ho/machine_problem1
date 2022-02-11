from nb_classifier import *
from custom_features import *

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

if __name__ == "__main__":
	if len(sys.argv) < 3:
		# sys.exit('please supply 2 command line arguments: [train filename] [test filename]')
		pass

	train_fname = sys.argv[1]
	test_fname = sys.argv[2]
	train_df = pd.read_csv(train_fname, names=header)
	test_df = pd.read_csv(test_fname, names=header)

	# ----------------------------------
	# Customization
	
	add_custom_features([train_df, test_df], all_params)

	my_params = [
		'gender', 
		'weight_kg', 
		# 'diastolic', 
		'grip_force', 
		'sit_and_bend_forward_cm', 
		'sit_up_count',
		'bp_class'
	]
	# my_weights = np.array([-3.72967159,  5.92218948,  3.1171761 ,  6.59952667,  4.32016449, 2.76731788])
	# my_weights = np.array([1.34873419, 1.54725775, 0.44241966, 3.20697321, 1.15853086, 0.67693897])
	# my_weights = np.array([ 0.48216685,  2.75703634,  4.44995824,  2.14060098,  1.10814467, -1.19870044])
	my_weights = np.ones(len(my_params))

	nb = NBClassifier()
	nb.train(train_df, print_time=False)

	# personal testing vs gradescope submission
	if len(sys.argv) == 3:
		nb.test(test_df, my_params, my_weights, print_pred=True, report=False)
	else:
		nb.test(test_df, my_params, my_weights, print_pred=False, report=True)

	# top10 = test_df[:10]
	# nb.test(top10, params, print_pred=True, report=False)
