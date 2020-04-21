from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from ml_utils import local_search
from sklearn.model_selection import cross_validate
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from utils import flatten


def seach_xgb_parameters(X_train, Y_train, X_valid, Y_valid):
    cv_folds = 3
    param_comb = 10
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 7],
    }
    model = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic', nthread=1)
    folder = StratifiedKFold(n_splits=param_comb, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='roc_auc',
                                       n_jobs=4, cv=folder.split(X_train, Y_train), verbose=3, random_state=1001)
    random_search.fit(X_train, Y_train)
    pass
    print(random_search.cv_results_)
    print(random_search.best_estimator_)


def local_search_xgb(X_train, Y_train, X_valid, Y_valid):
    init_params = {
        'subsample': 0.8,
        'colsample_bytree': 1.,
        'learning_rate': 0.1,
        'min_child_weight': 1.,
        'gamma': 0.1,
        'max_depth': 10,
        'n_estimators': 100
    }
    limits = {'subsample': (0., 1.), 'colsample_bytree': (0., 1.)}
    create_model = lambda params: XGBClassifier(objective='binary:logistic',  **params)
    score_function = lambda params: np.average(cross_validate(create_model(params), X_train, Y_train,
                                                              cv=10, n_jobs=5, verbose=0)['test_score'])
    best_params = local_search(init_params, score_function, limits=limits)
    best_model = create_model(best_params)
    best_model.fit(X_train, Y_train,)
    print("valid score = {}".format(best_model.score(X_valid, Y_valid)))
    return best_model


def simple_network(X_train, Y_train, X_valid, Y_valid, channels=200, layers=3, dropout=0.3):

    model = keras.Sequential(
        layers * [
            keras.layers.Dense(channels, activation='elu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
        ] + [
            keras.layers.Dense(1, activation='sigmoid')
        ]
    )
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000)
    model.fit(X_train, Y_train, validation_split=0.3, verbose=2, epochs=2000, callbacks=[es])
    train_acc = model.evaluate(X_train, Y_train, verbose=0)
    print('\nTrain accuracy:', train_acc)
    test_acc = model.evaluate(X_valid, Y_valid, verbose=2)

    print('\nTest accuracy:', test_acc)


def train_xgb_model(X_train, Y_train, X_valid, Y_valid):
    xgmodel = XGBClassifier(max_depth=13,
                            objective='binary:logistic',
                            gamma=0.1)
    xgmodel.fit(X_train, Y_train, verbose=True)
    train_score = xgmodel.score(X_train, Y_train)
    print("XGBoost train score: {}".format(train_score))
    valid_score = xgmodel.score(X_valid, Y_valid)
    print("XGBoost valid score: {}".format(valid_score))
    return xgmodel


def train_logistic(X_train, Y_train, X_valid, Y_valid):
    logmodel = LogisticRegression(solver='liblinear', C=3)
    logmodel.fit(X_train, Y_train)
    train_score = logmodel.score(X_train, Y_train)
    print("Logistic regression train score: {}".format(train_score))
    log_score = logmodel.score(X_valid, Y_valid)
    print("Logistic regression valid score: {}".format(log_score))
    return logmodel
