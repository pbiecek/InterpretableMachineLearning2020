import shap
from model import predict_treatment
import numpy as np
from model import calc_uplift
import dalex


def shapley_tree(model_predict, obs, dataset, column_names, plot_draw=False):
    explainer = shap.KernelExplainer(model_predict, shap.sample(dataset, 100))
    shap_values = explainer.shap_values(obs)
    if plot_draw:
        shap.waterfall_plot(explainer.expected_value, shap_values, feature_names=column_names)
    return shap_values, explainer.expected_value


def shapley_diff(model, obs, dataset, column_names, treatment_col, plot_draw=True):
    shap_t0, exp0 = shapley_tree(predict_treatment(model, treatment_col, 0), obs, dataset, column_names)
    shap_t1, exp1 = shapley_tree(predict_treatment(model, treatment_col, 1), obs, dataset, column_names)
    if plot_draw:
        shap.waterfall_plot(exp1 - exp0, shap_t1 - shap_t0, feature_names=column_names)
    return shap_t1 - shap_t0, exp1 - exp0


def shapley_importance_plot(model, dataset, column_names, treatment_col, sample_size=100):
    sample = np.random.randint(0, len(dataset), sample_size)
    values, expected = shapley_diff(model, dataset[sample], dataset, column_names, treatment_col, plot_draw=False)
    shap.summary_plot(values, dataset[sample], plot_type="bar", feature_names=column_names)


def shapley_variable_dependence_plot(model, dataset, column_names, treatment_col, pairs, sample_size=10):
    sample = np.random.randint(0, len(dataset), sample_size)
    values, expected = shapley_diff(model, dataset[sample], dataset, column_names, treatment_col, plot_draw=False)
    for pair in pairs:
        shap.dependence_plot(pair, values, dataset[sample])


def feature_importance_groups(model, X, group, column_names, treatment_col):
    uplift = calc_uplift(model, X, treatment_col)
    if group == "sleeping_dogs":
        group = X[uplift<0]
    elif group == "sure_things_and_lost_causes":
        group = X[(uplift >= 0) & (uplift <= 0.01)]
    elif group == "persuadables":
        group = X[uplift>0.01]
    return shapley_importance_plot(model, group, column_names, treatment_col)


def pdp_plot_uplift(model, X_train, Y_train, treatment_col):
  exp = dalex.Explainer(model, X_train, Y_train,
                        predict_function=lambda model, x: calc_uplift(model, x, treatment_col))
  partial = exp.model_profile(type='partial')
  partial.plot()


def ale_plot_uplift(model, X_train, Y_train, treatment_col):
  exp = dalex.Explainer(model, X_train, Y_train,
                        predict_function=lambda model, x: calc_uplift(model, x, treatment_col))
  ale = exp.model_profile(type='accumulated')
  ale.plot()
