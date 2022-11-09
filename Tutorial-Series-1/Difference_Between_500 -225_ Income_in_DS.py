# df.loc[0,cols] , 0-th row contain all questions , 246 columns

# Q1 age, Q2 gender, Q3 country, Q4 education, Q5 job title, Q10 income
cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q10'] 

def plot_distirubution(df, cols, normalize=True):
    # for axes iterate through each row & column
    # for unique value count for each cols use separate index
    # to add text use loop on axes[row][col].patches

quenstion_dict # map which question belon to which part , size : 247
# 'Time from Start to Finish (seconds)': 'Duration (in seconds)'
# 'Q1': 'What is your age (# years)?'
# 'Q34_Part_12' : 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Other'
# 'Q34_OTHER_TEXT': 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Other - Text'
# 'compensation_num' : 'What is your current yearly compensation (approximate USD)?'

#  combine cols_1 + cols_2 in a list
def group_cols(df): 
	# separate columns with _ in it's name -> col_part
	# columns with no _ in it's name -> cols_1
	# temp_group.value_counts() -> corresponding to each cols_1 value of col_part
	#    						   with same question      
	# cols_2 -> column names from temp_group with group size > 1
       
questions = group_cols(df) #size : 37 
df_survey_result = df1.loc[1:, :] # all responses columns
target_col = 'compensation_num_group'
def categorical_distribution_diff(x, y):
	# x -> all 'low' values for target_col in df_survey_result
	# y -> all 'high' values for target_col in df_survey_result
	# x_counts -> for each of the questions get unique response count
	# y_counts -> same as above but for 'high' group

def find_distribution_diff(df1,questions, target_col):
	# score_np -> size : 37 (same as questions) , store sum of categorical
	               # distribution difference
	# sort_index -> size : 37 , 
					# value : [36  8  0 11 13  5  7  9  1 15 10  6 14  
					#			2 12  3  4 17 18 21 34 19 28 30
 					#			22 32 31 33 20 16 24 29 35 25 26 27 23]
 	# score_cols -> for each question : (questions[order], score_np[order])
 					# value : 'Q21': (['Q21_Part_1', 'Q21_Part_2', 'Q21_Part_3', 'Q21_Part_4', 'Q21_Part_5'], 5.0)
 					#	'compensation': (['compensation_num', 'compensation_num_group'], 8809.0)
 					#  'Q3': ('Q3', 6302.0) -> this is bases on similar response group
 					# size : 36


def long_sentences_seperate(sentence, step=10):
	# input -> x_list = ['Q4', 'Q15', 'Q15', 'Q15', 'Q15']
			#  y_list = ['Q2', 'Q1', 'Q4', 'Q5', 'Q19']
	# split by " " and '\n'

x_list = ['Q4', 'Q15', 'Q15', 'Q15', 'Q15']
y_list = ['Q2', 'Q1', 'Q4', 'Q5', 'Q19']
x_values = temp_df[x].unique()

def plot_point_salary(df, target='', country='all'):
	# temp_df -> size : (12497, 248)
	# ttemp_df -> size : for 15 different plots , 15 different (row,col) based
				# on target_value=['low', 'medium', 'high']
	for x,y in zip(x_list,y_list):
		for target_value in ['low', 'medium', 'high']:
			for value_name_x in x_values:
				# counts_y -> for matching x_values with x , count overlapping y
				for value_name_y in counts_y.index:
					# condition_1 -> matching both x & y-th row value_name_x & value_name_y
					# condition_2 -> condition_1 & target_value with target-th row
					# temp_df & counts_list -> store value_name_y (till not sure)

					# size : counts_list [75 219 146 205 ]
					# value (= counts_list[75]): [1634, 401, 19, 3, 165, 46, 1, 1435, 240, 18, 3, 426, 130, 9, 1, 162, 24, 6, 70, 12, 10, 1, 44, 10, 3, 1589, 240, 20, 1, 134, 15, 1, 1, 756, 105, 5, 505, 103, 11, 3, 95, 6, 3, 52, 4, 2, 33, 4, 1669, 261, 24, 7, 89, 7, 2, 1, 698, 87, 11, 3, 756, 122, 15, 2, 107, 7, 3, 1, 26, 5, 1, 1, 28, 2, 1]

					# size : temp_df (12497, 249)


		# temp_df[x],temp_df[y] -> apply long sentence separate (12497,) * 5
									# i.e 5 = len(x_list),len(y_list)


temp_df['Q15'] = temp_df['Q15'].fillna('None')
temp_df['Q11'] = temp_df['Q11'].fillna('None')

x_list = ['Q2', 'Q3', 'Q5']
y_list = ['Q4', 'Q1', 'Q15']
z_list = ['Q5', 'Q6', 'Q11']
title_list = ['Genders', 'Countries', 'Job Titles', 'Courses']

def plot_treemap_salary(df, target='', country='all'):
	for title, x, y, z in zip(title_list, x_list, y_list, z_list):
		x_values = temp_df[x].unique()
        z_values = temp_df[z].unique()
        # target_value -> value_name_z -> value_name_x -> value_name_y 
        # 3 conditions for temp_df & counts_list