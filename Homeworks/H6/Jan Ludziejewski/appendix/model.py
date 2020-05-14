from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from ml_utils import local_search
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from pylift.eval import UpliftEval
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd


def calc_uplift(model, x, treatment_col):
    return predict_treatment(model, treatment_col, 1)(x) - predict_treatment(model, treatment_col, 0)(x)


def predict_treatment(model, treatment_col, c):
    def pred(x):
        if type(x) is pd.DataFrame:
            x = x.to_numpy()
        x = x.copy()
        x[:, treatment_col] = c
        return model.predict_proba(x)[:, 1]
    return pred


def check_acc_diff(model, name, X_train, Y_train, X_valid, Y_valid):
    print("{} train acc: {}".format(name, model.score(X_train, Y_train)))
    print("{} valid acc: {}".format(name, model.score(X_valid, Y_valid)))


def check_uplift_diff(model, name, X_train, Y_train, X_valid, Y_valid, treatment_col):
    print("{} train uplift score = {}".format(name, evaluate_uplift(model, X_train, Y_train, treatment_col, plot=True)))
    print("{} valid uplift score = {}".format(name, evaluate_uplift(model, X_valid, Y_valid, treatment_col, plot=True)))


def evaluate_uplift(model, x, y, treatment_col, plot=False):
    uplift = calc_uplift(model, x, treatment_col)
    upe = UpliftEval(x[:, treatment_col], y, uplift)
    if plot:
        upe.plot(show_theoretical_max=True, show_practical_max=True, show_no_dogs=True, show_random_selection=True)
        plt.plot()

    return upe.q2_cgains


def get_cv_score(create_model, X, Y, treat_col, cv=6):
    scores = []
    for i in range(cv):
        xt, xv, yt, yv = train_test_split(X, Y, test_size=0.3, stratify=Y)
        model = create_model()
        model.fit(xt, yt)
        scores.append(evaluate_uplift(model, xv, yv, treat_col))
    return np.average(scores)


def local_search_svm(X_train, Y_train, X_valid, Y_valid, treatment_col, just_get_model=False, plot=False):
    init_params = {
        'C': 10.,
        'gamma': 0.1
    }
    limits = {}
    create_model = lambda params: SVC(kernel='rbf', probability=True, **params)
    score_function = lambda params: get_cv_score(lambda: create_model(params), X_train, Y_train, treatment_col, cv=6)
    best_model = create_model(init_params if just_get_model else
                              local_search(init_params, score_function, limits=limits))
    best_model.fit(X_train, Y_train)
    print("train score = {}".format(evaluate_uplift(best_model, X_train, Y_train, treatment_col, plot=plot)))
    print("valid score = {}".format(evaluate_uplift(best_model, X_valid, Y_valid, treatment_col, plot=plot)))
    return best_model


def local_search_xgb(X_train, Y_train, X_valid, Y_valid, treatment_col, just_get_model=False, print_score=False, init_params=None):
    init_params = init_params or {
                   'subsample': 0.775205270351832, 'colsample_bytree': 0.25,
                   'learning_rate':  0.7474227529937205, 'min_child_weight':  0.3192759517420326,
                   'gamma': 0.05306043438668296, 'max_depth': 5, 'n_estimators': 12
}
    limits = {'subsample': (0., 1.), 'colsample_bytree': (0., 1.)}
    create_model = lambda params: XGBClassifier(objective='binary:logistic',  **params)
    score_function = lambda params: get_cv_score(lambda: create_model(params), X_train, Y_train, treatment_col, cv=12)
    best_model = create_model(init_params if just_get_model else
                              local_search(init_params, score_function, limits=limits))
    best_model.fit(X_train, Y_train)
    if print_score:
        print("train score = {}".format(evaluate_uplift(best_model, X_train, Y_train, treatment_col)))
        print("valid score = {}".format(evaluate_uplift(best_model, X_valid, Y_valid, treatment_col)))
    return best_model


def simple_network(X_train, Y_train, X_valid, Y_valid, channels=30, layers=3, epochs=5, verbosity=0):
    def create_model():
        model = keras.Sequential(
            layers * [
                keras.layers.Dense(channels, activation='relu'),
                tf.keras.layers.BatchNormalization(),
            ] + [
                keras.layers.Dense(1, activation='sigmoid')
            ]
        )
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                               'accuracy', tf.keras.metrics.AUC()])
        return model
    model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=verbosity)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbosity, patience=50)
    model.fit(X_train, Y_train, validation_split=0.3, verbose=verbosity, epochs=epochs, callbacks=[es])
    if verbosity > 0:
        train_acc = model.score(X_train, Y_train, verbose=0)
        print('\nTrain accuracy:', train_acc)
        test_acc = model.score(X_valid, Y_valid, verbose=2)
        print('\nTest accuracy:', test_acc)
    return model


def train_xgb_model(X_train, Y_train, X_valid, Y_valid, print_score=False):
    xgmodel = XGBClassifier(max_depth=13,
                            objective='binary:logistic',
                            gamma=0.1)
    xgmodel.fit(X_train, Y_train, verbose=True)
    if print_score:
        train_score = xgmodel.score(X_train, Y_train)
        print("XGBoost train score: {}".format(train_score))
        valid_score = xgmodel.score(X_valid, Y_valid)
        print("XGBoost valid score: {}".format(valid_score))
    return xgmodel


def train_logistic(X_train, Y_train, X_valid, Y_valid, print_score=False):
    logmodel = LogisticRegression(solver='liblinear', C=3)
    logmodel.fit(X_train, Y_train)
    if print_score:
        train_score = logmodel.score(X_train, Y_train)
        print("Logistic regression train score: {}".format(train_score))
        log_score = logmodel.score(X_valid, Y_valid)
        print("Logistic regression valid score: {}".format(log_score))
    return logmodel
