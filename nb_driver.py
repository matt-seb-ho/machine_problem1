from nb_classifier import *

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

def classify_bp(row):
	dia, sy = row['diastolic'], row['systolic']
	if sy < 120 and dia < 80:
		return 'normal'
	if sy < 130 and dia < 80:
		return 'elevated'
	if sy < 140 or (dia >= 80 and dia < 90):
		return  'stage1'
	return 'hypertension'

def add_bp_class(df):
	df['bp_class'] = df.apply(classify_bp, axis=1)

def add_bmi(df):
	df['bmi'] = 10000 * df['weight_kg'] / (np.square(df['height_cm']))

if __name__ == "__main__":
	if len(sys.argv) < 3:
		sys.exit('please supply 2 command line arguments: [train filename] [test filename]')

	train_fname = sys.argv[1]
	test_fname = sys.argv[2]
	train_df = pd.read_csv(train_fname, names=header)
	test_df = pd.read_csv(test_fname, names=header)

	# ----------------------------------
	# Customization
	
	add_bp_class(train_df)
	add_bp_class(test_df)
	add_bmi(train_df)
	add_bmi(test_df)

	my_params = [
		'gender', 
		# 'weight_kg', 
		'diastolic', 
		'grip_force', 
		'sit_and_bend_forward_cm', 
		'sit_up_count',
		# 'bp_class',
		'bmi'
	]

	nb = NBClassifier()
	nb.train(train_df, print_time=False)

	# personal testing vs gradescope submission
	if len(sys.argv) == 3:
		nb.test(test_df, my_params, print_pred=True, report=False)
	else:
		nb.test(test_df, my_params, print_pred=False, report=True)

	# top10 = test_df[:10]
	# nb.test(top10, params, print_pred=True, report=False)
