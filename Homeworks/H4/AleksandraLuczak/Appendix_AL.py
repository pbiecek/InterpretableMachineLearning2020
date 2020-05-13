import pandas as pd
import numpy as np
from ceteris_paribus.explainer import explain
from ceteris_paribus.profiles import individual_variable_profile
from ceteris_paribus.plots.plots import plot_notebook

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier


def prepare_data(path='../data/heloc_dataset_v1.csv'):
    data = pd.read_csv(path)
    data['RiskPerformance'] = np.where(data.RiskPerformance=='Bad',1,0)
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('RiskPerformance', axis =1), 
        data.RiskPerformance, 
        test_size=.33, 
        random_state=42)
    seletor_f_classif = SelectKBest(k='all')
    seletor_f_classif.fit_transform(X_train, y_train)
    column_list = pd.DataFrame({
    'column':X_train.columns, 
    'score': seletor_f_classif.scores_, 
    'p_value':seletor_f_classif.pvalues_
    }).query("score>10").column.values
    return column_list

def split_data(path='../data/heloc_dataset_v1.csv'):
    data = pd.read_csv(path)
    column_list = prepare_data()
    return train_test_split(
        data[column_list], 
        data.RiskPerformance, 
        test_size=.33, 
        random_state=42)

def load_model_lin():
    X_train, X_test, y_train, y_test=split_data()
    clf = LogisticRegression(
        random_state=123,
    max_iter = 200,
    solver = 'liblinear'
    )
    clf.fit(X_train, y_train)
    return clf

def load_model_nn():
    X_train, X_test, y_train, y_test=split_data()
    clf_nn = MLPClassifier(alpha=1, max_iter=1000)
    clf_nn.fit(X_train, y_train)
    return clf_nn

def create_cp(model, label, idx, path='../data/heloc_dataset_v1.csv'):
    data = pd.read_csv(path)
    column_list = prepare_data()
    explainer = explain(model, data=data[column_list], y=data.RiskPerformance, label=label,
    predict_function=lambda X: model.predict_proba(X)[::, 1])
    return individual_variable_profile(explainer, data[column_list].loc[idx], data.loc[idx,'RiskPerformance'])

def create_cp_different_profiles():
    model_lin = load_model_lin()
    TP = create_cp(model_lin, "TruePositive",7585)
    FP = create_cp(model_lin, "FalsePositive",3785)
    TN = create_cp(model_lin, "TrueNegative",1039)
    FN = create_cp(model_lin, "FalseNegative",6409)
    plot_notebook(TP, FP,TN, FN, selected_variables=["MSinceMostRecentTradeOpen"],aggregate_profiles='mean')

def create_cp_different_models():
    model_lin = load_model_lin()
    model_nn = load_model_nn()
    FN = create_cp(model_lin, "linear",6409)
    FN2 = create_cp(model_nn, "NeuralNet",6409)
    plot_notebook(FN, FN2, selected_variables=['MSinceOldestTradeOpen'],aggregate_profiles='mean')
    plot_notebook(FN, FN2, selected_variables=["AverageMInFile"],aggregate_profiles='mean')
    plot_notebook(FN, FN2, selected_variables=['MSinceMostRecentTradeOpen'],aggregate_profiles='mean')

