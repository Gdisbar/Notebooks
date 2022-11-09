skew_cols : ['A0T0G0C10', 'A0T0G1C9', 'A0T0G2C8', 'A0T0G8C2', 'A0T0G9C1', 
'A0T0G10C0', 'A0T1G1C8', 'A0T1G8C1', 'A0T1G9C0', 'A0T2G0C8', 'A0T2G8C0', 
'A0T10G0C0', 'A1T0G0C9', 'A1T0G8C1', 'A1T1G0C8', 'A1T1G8C0', 'A2T0G0C8', 
'A2T0G8C0', 'A10T0G0C0']

ncols = 2
nrows = round(len(skew_cols) / ncols) # value : 10

#embedding on train[:,:-1] ,i.e without 'target column' -> on each row
embedding # size : train.shape[0] * 2 i.e 123993 * 2 

colors = pd.factorize(train.loc[:, 'target']) 
colors[0] 
# size : train.shape[0] i.e 123993
# value : array([0, 1, 1, ..., 8, 1, 5]) 
color[1] 
# size : 10 , no. of unique values in 'target' column
# value : ['Streptococcus_pyogenes', 'Salmonella_enterica', 'Enterococcus_hirae',
#        'Escherichia_coli', 'Campylobacter_jejuni', 'Streptococcus_pneumoniae',
#        'Staphylococcus_aureus', 'Escherichia_fergusonii',
#        'Bacteroides_fragilis', 'Klebsiella_pneumoniae']

# colors_dict : 'int' to 'target' mapping
for color_key in colors_dict.keys():
    indexs = colors[0] == color_key
    idx.append(indexs)
    temp_embedding = embedding[indexs, :]
    embed.append(temp_embedding)

idx  # size : 10 * 123993 , i.e train.shape[0]
idx[9] # value : array([False, False, False, ..., False, False, False])

embed # size : 10 * variable no of columns
embed[0] 
# size : 12406 
# value : array([[-11.152245 ,  16.00027  ],
	       # [  3.9808948,  11.574094 ],
	       # [ -2.747582 ,  18.810406 ],
	       # ...,
	       # [ 13.625458 ,  14.010201 ],
	       # [ -2.1082287,  18.165407 ],
	       # [ -2.5992832,  18.769602 ]], dtype=float32)

embed[9]
# size  : 12420 
# value : array([[-10.68038  ,  15.762704 ],
		       # [ -5.6278176,   7.767241 ],
		       # [ -3.9533765,   5.74988  ],
		       # ...,
		       # [ -2.8760586,   5.1771235],
		       # [ -5.970015 ,   6.2508726],
		       # [ -9.209835 ,  -2.3163419]], dtype=float32)

embed[9][:,0]
# size : 12420
# array([-10.68038  ,  -5.6278176,  -3.9533765, ...,  -2.8760586,
#         -5.970015 ,  -9.209835 ], dtype=float32)

embed[9][:,1]
# size : 12420
# value : array([15.762704 ,  7.767241 ,  5.74988  , ...,  5.1771235,  6.2508726,
#        -2.3163419], dtype=float32)



p_values_target_list # size : 286 * 10 , here no of neumerical columns = 286

def p_value_warning_background(cell_value):
    highlight = 'background-color: lightcoral;'
    default = ''
    if cell_value > 0.05:
            return highlight
    return default

p_values_df.style.applymap(p_value_warning_background)

important_dict # under each target value : list of neumeric columns

upper = train[numeric_cols].corr().where(np.triu(np.ones(all_feature_corr.shape), k=1).astype(np.bool))
# 286 * 286 , upper triangle of corr matrix , do contain nan + add 'target'
targets_train = targets.merge(train.drop('target', axis=1), left_index=True, right_index=True)
all_target_corr = targets_train.corr()

===============================================================================
## TPS-Feb22, Pycaret-Model Comparisons
===============================================================================
Blending
-------------
N = 2
include = ['nb', 'ridge', 'rf', 'et', 'dt', 'lr', 'qda', 'lda', 'lightgbm']
include = ['rf', 'et']

top = compare_models(sort = 'Accuracy', n_select = N, include = include)
tuned_top = [tune_model(i, optimize = 'accuracy', 
				choose_better=True, n_iter=100) for i in top]

blend = blend_models(top, optimize='Accuracy')

predict_model(blend);
final_blend = finalize_model(blend)
# 10 * no of each predicted class in dataset
plot_model(final_blend, plot='error') 
plot_model(final_blend, plot = 'confusion_matrix')

Ensembling
-------------

best = compare_models(sort = 'Accuracy', include = include)
tuned = tune_model(best, optimize = 'accuracy', choose_better=True, n_iter=100)
ensemble = ensemble_model(best, method='Bagging')

Stacking
-------------
top = compare_models(sort = 'Accuracy', n_select = N, include = include)
stack = stack_models(top, optimize='Accuracy')


assert(len(test.index)==len(unseen_predictions))
sub = pd.DataFrame(list(zip(sub.row_id, unseen_predictions.Label)),columns = ['row_id', 'target'])
sub.to_csv('submission_stack.csv', index = False)
