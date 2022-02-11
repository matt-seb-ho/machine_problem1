from nb_classifier import * 

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

def classify_bmi(row):
	bmi = row['bmi']
	if bmi < 18.5:
		return 'under'
	elif bmi < 25:
		return 'normal'
	elif bmi < 30:
		return 'over'
	return 'obese'

def add_bmi_class(df):
	df['bmi_class'] = df.apply(classify_bmi, axis=1)

def add_custom_features(dfs, all_p):
    for df in dfs:
        add_bmi(df)
        add_bmi_class(df)
        add_bp_class(df)
    all_p.extend(['bmi', 'bmi_class', 'bp_class'])
