# Random Forest Baseline Model
=====================================
CAT_FEATURES = ['Sex', 'Embarked']
NUM_FEATURES = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
IGN_FEATURES = ['PassengerId', 'Name', 'Ticket', 'Cabin']
clf = setup(...)
compare_models(sort='Accuracy')
baseline_model = create_model('rf')
baseline_preds = predict_model(baseline_model, raw_score=True)
baseline_scores = pull()

=============================================================================
# Features Importance
=============================================================================
# titanic_train_FeaEng , after doing feature engineering on train data

feature_score_dict = {}

for index, feature in enumerate(new_features):
    old_features_temp = old_features.copy()
    old_features_temp.append(feature)
    titanic_train_FeaEng_temp = titanic_train_FeaEng[old_features_temp].copy()
    
    clf = setup(...)
    
    baseline_model = create_model('rf')
    scores = pull()
    feature_score_dict[feature] = scores



===========================================================================
# LightAutoML
===========================================================================
# from lightautoml.automl.presets.tabular_presets import TabularAutoML, 
# TabularUtilizedAutoML
# from lightautoml.dataset.roles import DatetimeRole
# from lightautoml.tasks import Task

# import torch

# N_THREADS = 4 # threads cnt for lgbm and linear models
# N_FOLDS = 5 # folds cnt for AutoML
# RANDOM_STATE = 42 # fixed random state for various reasons
# TEST_SIZE = 0.2 # Test size for metric check
# TIMEOUT = 300 # Time in seconds for automl run

# np.random.seed(RANDOM_STATE)
# torch.set_num_threads(N_THREADS)

# def acc_score(y_true, y_pred, **kwargs):
#     return accuracy_score(y_true, (y_pred > 0.5).astype(int), **kwargs)

# def f1_metric(y_true, y_pred, **kwargs):
#     return f1_score(y_true, (y_pred > 0.5).astype(int), **kwargs)

# task = Task('binary', metric = acc_score)

# roles = {
#     'target': 'Survived',
#     'drop': ['Passengerid', 'Name', 'Ticket'],
# }

for train_index, test_index in skf.split(titanic_train, titanic_train['Survived']):
    # X_train, X_test = titanic_train.loc[train_index, :], titanic_train.loc[test_index, :]
    # y = X_test['Survived']
    # X_test.drop(['Survived'], axis=1, inplace=True)
    
    automl = TabularUtilizedAutoML(task = task, 
                timeout = TIMEOUT,
                cpu_limit = N_THREADS,
                general_params = {'use_algos': [['linear_l2', 'lgb', 'lgb_tuned']]},
                reader_params = {'n_jobs': N_THREADS})
    automl.fit_predict(X_train, roles = roles)
    
#     test_pred = automl.predict(X_test)
#     test_pred = (test_pred.data[:, 0] > 0.5).astype(int)
#     acc_list.append(acc_score(y, test_pred))
# lightautoml_acc_score = sum(acc_list) / n_fold

============================================================================
# H2O AutoML
=============================================================================
acc_list = []
for train_index, test_index in skf.split(titanic_train, titanic_train['Survived']):
    # X_train, X_test = titanic_train.loc[train_index, :], titanic_train.loc[test_index, :]
    # y = X_test['Survived'].astype(int)
    # X_test.drop(['Survived'], axis=1, inplace=True)
    
    train_hf = h2o.H2OFrame(X_train.copy())
    test_hf = h2o.H2OFrame(X_test.copy())
    feature_columns = X_train.drop(['Survived', 'PassengerId'], axis=1).columns
    
    aml = H2OAutoML(
        seed=2022, 
        max_runtime_secs=100,
        nfolds = 3,
        exclude_algos = ["DeepLearning"]
    )
    
    aml.train(
        x=list(feature_columns), 
        y='Survived', 
        training_frame=train_hf
    )
    
    # test_pred = aml.predict(test_hf)
    # test_pred = test_pred.as_data_frame()
    # test_pred['test_pred_int'] = (test_pred[['predict']] > 0.5)
    # y_pred = test_pred['test_pred_int'].astype(int)
    # h2o_acc_score = accuracy_score(y, y_pred)
    # acc_list.append(h2o_acc_score)
#h2o_tautoml_acc_score = sum(acc_list) / n_fold