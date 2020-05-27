import pandas as pd
import numpy as np
import pickle
from dalex.dataset_level import AggregatedProfiles
from dalex import Explainer


def prepare_data_to_model(path='../data/survival.csv'):
    data = pd.read_csv(path, index_col = 0).drop(['id','study','etype','status'], axis = 1)
    data['treatment'] = np.where(data['rx']=='Obs', 0, np.where(data['rx']=='Lev',1,2))
    data.drop('rx', axis=1, inplace = True)
    data = data.fillna(-1)
    return data.drop(['treatment','time'], axis=1), data['treatment'], data['time']

def plot_pdp_ald(model, X, y, treatment="Levamisole", plot_type = 'partial'):
    if treatment=="Levamisole":
        treat=0
    else:
        treat=1
    exp = Explainer(model, X, y, verbose=False, predict_function= (lambda model, x: model.predict(x)[:,treat]))
    pdp = exp.model_profile(
        N=300,
        variables= ['age','nodes'],
        type=plot_type)
    pdp.plot(title=f'Treatment {treatment}')