import shap
from model import predict_treatment
import numpy as np


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
