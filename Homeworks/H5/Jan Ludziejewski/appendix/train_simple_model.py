import shap
from data import X_train, Y_train, X_valid, Y_valid, treatment_col, column_names
from model import train_xgb_model, train_logistic, simple_network, evaluate_uplift
from pylift.eval import UpliftEval
import matplotlib

xgbmodel = train_xgb_model(X_train, Y_train, X_valid, Y_valid)

#upev = UpliftEval(X_valid[:, treatment_col], xgbmodel.predict_proba(treatment_col))
#nmodel = simple_network(X_train, Y_train, X_valid, Y_valid)
#logmodel = train_logistic(X_train, Y_train, X_valid, Y_valid)


# plot importance
#explainer = shap.TreeExplainer(xgbmodel)
#xg_shap_values = explainer.shap_values(X_train.iloc[:100, :])
#shap.force_plot(explainer.expected_value, xg_shap_values[0, :], X_train.iloc[0, :], matplotlib=True)
#shap.summary_plot(xg_shap_values, X_train.iloc[:100, :], feature_names=list(X_train.columns))

print("train gain={}".format(evaluate_uplift(xgbmodel, X_train, Y_train, treatment_col, plot=True)))
print("valid gain={}".format(evaluate_uplift(xgbmodel, X_valid, Y_valid, treatment_col, plot=True)))