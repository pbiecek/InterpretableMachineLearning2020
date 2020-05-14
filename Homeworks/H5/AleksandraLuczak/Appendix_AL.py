import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

def prepare_data():
    df = pd.read_csv('../data/survival.csv', index_col = 0).drop(['id','study','etype'], axis = 1)
    df['treatment'] = np.where(df['rx']=='Obs', 0, np.where(df['rx']=='Lev',1,2))
    df.drop('rx', axis=1, inplace = True)
    df.differ = df.differ.fillna(-1)
    df.nodes = df.nodes.fillna(0)
    X = df.drop(['treatment','time'], axis=1)
    y = df['time']
    treatment = df['treatment']
    return X, treatment, y


def plot_importance(model):

    X, treatment, y = prepare_data()
    model.estimate_ate(X, treatment, y)
    model_tau = model.fit_predict(X, treatment, y,verbose=False)
    importance_dict = model.get_importance(X=X, 
                tau=model_tau, 
                method='permutation', 
                features=X.columns, 
                random_state = 123)
    for group, series in importance_dict.items():
        if group ==1:
            plt.subplot(1, 2, 1)
            plt.title("Treatment Levamisole")
        else:
            plt.subplot(1, 2, 2)
            plt.title("Treatment Levamisole i 5-FluoroUracyl")    
        series.sort_values().plot(kind='barh', figsize=(15, 6))
    return importance_dict

def train_and_save_model(model):
    X, treatment, y = prepare_data()
    model.estimate_ate(X, treatment, y)
    with open('xl.pickle', 'wb') as file:
        pickle.dump(model, file)    
    model_tau = model.fit_predict(X, treatment, y,verbose=False)

    with open('xl_tau.pickle', 'wb') as file:
        pickle.dump(model_tau, file)

def plotx_importance():

    model= pickle.load(open('xl.pickle', 'rb'))
    model_tau = pickle.load(open('xl_tau.pickle', 'rb'))
    X, treatment, y = prepare_data()
    importance_dict = model.get_importance(X=X, 
                tau=model_tau, 
                method='permutation', 
                features=X.columns, 
                random_state = 123)
    for group, series in importance_dict.items():
        if group ==1:
            plt.subplot(1, 2, 1)
            plt.title("Treatment Levamisole")
        else:
            plt.subplot(1, 2, 2)
            plt.title("Treatment Levamisole i 5-FluoroUracyl")    
        series.sort_values().plot(kind='barh', figsize=(15, 6))
    return importance_dict







