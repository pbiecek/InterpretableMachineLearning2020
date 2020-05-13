import shap
shap.initjs()
from data import X_train, Y_train, X_valid, Y_valid, treatment_col
from model import train_xgb_model, train_logistic, local_search_xgb, local_search_svm


best_model = local_search_xgb(X_train, Y_train, X_valid, Y_valid, treatment_col, just_get_model=True)
pass
#best_model = seach_xgb_parameters(X_train, Y_train, X_valid, Y_valid)

{'subsample': 0.03375205270351832, 'colsample_bytree': 0.017281050984201383, 'learning_rate': 0.007474227529937205, 'min_child_weight': 0.016192759517420326, 'gamma': 0.0005306043438668296, 'max_depth': 4, 'n_estimators': 12}