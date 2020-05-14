from model import train_xgb_model, train_logistic
from data import train, X_train, Y_train, X_valid, Y_valid, column_names, treatment_col
import numpy as np
from ceteris_paribus.explainer import explain
from ceteris_paribus.profiles import individual_variable_profile
from ceteris_paribus.plots.plots import plot
from IPython.core.display import display, HTML
from IPython.display import IFrame

#print("Models: ")
count = 0

xgmodel = train_xgb_model(X_train, Y_train, X_valid, Y_valid)
#print("\n")
logmodel = train_logistic(X_train, Y_train, X_valid, Y_valid)
#print("\n")

print("Dataset bias: {}% of positive answers \n\n\n".format(np.average(Y_train)*100))

train.info()

xg_predicted = xgmodel.predict(X_valid)
lin_predicted = logmodel.predict(X_valid)


false_negatives = np.where(np.logical_and(xg_predicted != Y_valid, Y_valid == 0))


def explain_wrap(index, columns):
    print("DOING {}".format(index))
    global count
    x, y = X_valid[index], Y_valid[index]
    explainer_xgb = explain(xg_predicted, data=X_train, y=Y_train, label="XGBoost model",
        predict_function=lambda X: xgmodel.predict_proba(X.to_numpy())[::, 1], variable_names=column_names)
    explainer_linear = explain(lin_predicted, data=X_train, y=Y_train, label="Logistic model",
        predict_function=lambda X: logmodel.predict_proba(X.to_numpy())[::, 1], variable_names=column_names)
    cp_xgb = individual_variable_profile(explainer_xgb, x, y)
    cp_lin = individual_variable_profile(explainer_linear, x, y)
    plot(cp_xgb, cp_lin, selected_variables=columns, destination="browser", show_observations=False)
    #IFrame(src="./_plot_files/plots{}.html".format(count), width=700, height=600)
    #with open("_plot_files/plots{}.html".format(count), 'r') as myfile:
#        display(HTML(myfile.read()))
    count += 1


first_part = ["AGE", "TREATMENT", "N_OPEN_REV_ACTS", "D_REGION_A"]
second_part = ["MRTG_1_MONTHLY_PAYMENT", "N_DISPUTED_ACTS", "TOT_HI_CRDT_CRDT_LMT", "HI_RETAIL_CRDT_LMT"]
third_part = ["PREM_BANKCARD_CRED_LMT", "AVG_BAL_ALL_FNC_REV_ACTS", "M_SNCOLDST_OIL_NTN_TRD_OPN"]


train.head()


#explain_wrap(11, first_part)
#explain_wrap(11, second_part)
#explain_wrap(11, third_part)

#explain_wrap(2, first_part)
#explain_wrap(2, second_part)
#explain_wrap(2, third_part)

#explain_wrap(false_negatives[0][0], first_part)
#explain_wrap(false_negatives[0][0], second_part)
#explain_wrap(false_negatives[0][0], third_part)