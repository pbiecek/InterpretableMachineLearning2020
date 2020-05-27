import shap
shap.initjs()
from data import X_train, Y_train, X_valid, Y_valid, treatment_col
from model import train_xgb_model, train_logistic, local_search_xgb, local_search_svm, calc_uplift, check_uplift_diff
from explanations import pdp_plot_uplift, ale_plot_uplift
import numpy as np

best_model = local_search_xgb(X_train, Y_train, X_valid, Y_valid, treatment_col, just_get_model=True)
check_uplift_diff(best_model, "best", X_train, Y_train, X_valid, Y_valid, treatment_col)

pdp_plot_uplift(best_model, X_train, Y_train, treatment_col)
ale_plot_uplift(best_model, X_train, Y_train, treatment_col)

uplift = calc_uplift(best_model, X_valid, treatment_col)

print("{} -> {} (avg={} var={}) (<0) = {} (>0) = {}".format(
    uplift.min(), uplift.max(), np.average(uplift), np.var(uplift), np.sum(uplift < 0), np.sum(uplift > 0)))


best_params = {
                   'subsample': 0.775205270351832, 'colsample_bytree': 0.17281050984201383,
                   'learning_rate': 1, 'min_child_weight': 0.016192759517420326,
                   'gamma': 0.05306043438668296, 'max_depth': 5, 'n_estimators': 12
}

wide_params = {
                   'subsample': 0.775205270351832, 'colsample_bytree': 0.25,
                   'learning_rate':  0.7474227529937205, 'min_child_weight':  0.3192759517420326,
                   'gamma': 0.05306043438668296, 'max_depth': 5, 'n_estimators': 12
}