model = LGBMClassifier(**params)  
model.fit(train_x,train_y,eval_set=[(test_x,test_y)],eval_metric='auc', early_stopping_rounds=300, verbose=False)
preds = model.predict_proba(test_x)[:,1] # class,pred_proba
auc = roc_auc_score(test_y, preds)
# optuna use case
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
study.trials # finished trials
study.best_value
plot_optimization_history(study)
optuna.visualization.plot_param_importances(study) # Feature Importance
# using gpu with optuna
lgb_params = study.best_trial.params
lgb_params['device'] = "gpu"
lgb_params['cat_feature'] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
                             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 
                             53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
lgb_params['n_jobs'] = -1 
lgb_params['boosting_type'] =  'gbdt'
# stratified/level wise separation K-Fold
NFOLDS = 20
folds = StratifiedKFold(n_splits=NFOLDS, random_state=42, shuffle=True)
predictions = np.zeros(len(X_test))
for fold, (train_index, test_index) in enumerate(folds.split(X, y)):
    print("--> Fold {}".format(fold + 1))
    
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    lgb_model = LGBMClassifier(**lgb_params).fit(X_train, y_train, 
                                                  eval_set=[(X_valid, y_valid)], 
                                                  eval_metric='auc', 
                                                  early_stopping_rounds=300, verbose=0)
    
    y_preds = lgb_model.predict_proba(X_valid)[:,1]
    predictions += lgb_model.predict_proba(X_test)[:,1] / folds.n_splits 
    
    print(": LGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, y_preds, average="micro")))
