# feature_list -> categorical variabele with <= 30 unique value
# including target with categorical -> data_categorical_pd[:][:len(train_y)]
# label encode columns -> present both in train & validation set
# cat10_cut_values_list -> from bad columns take those that has value > th
# store them in a list -> add this dataframe (tmp) -> data_categorical_temp_pd[:][:len(train_y)], train_y
# train,test split

Feature Engineering
--------------------
divide feature based on their unique count # for categorical
combine encoded categorical features into -> data_numerical_FeaEng_pd 
'cat9cat2T' -> 'cat9'+'cat2', 'cat15cat17M' -> 'cat15'*'cat17'
#from previous steps
'cat15_16TotalTotal' -> 'cat15cat16T'+ 'cat15cat17T'+'cat15cat18T'
						+'cat16cat17T'+'cat16cat18T'+'cat17cat18T'
'cat11_14MulTotal' -> 'cat0cat11M'+'cat0cat12M'+... # same just take 'M' ones

# for numerical features -> use MinMaxScaler
norm_list -> MinMaxScaler for singel column , 
norm_list+1 -> to make data +ve
data_numerical_TR_pd[feature] -> scale down with fitted_lambda


n_rows = round(data_numerical_TR_pd.shape[1] / 4)
n_rows = n_rows - 1
fig, axs = plt.subplots(nrows=n_rows, ncols=4, figsize=(20, 10))
# skew_feature_list numerical
skew_feature_list -> ['cont0', 'cont2', 'cont5', 'cont7', 'cont8', 'cont9', 'cont10']
