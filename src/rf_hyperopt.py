import numpy as np
import pandas as pd
from functools import partial
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from hyperopt import hp, fmin, tpe, Trials
from sklearn.model_selection import KFold, cross_val_score
from hyperopt.pyll.base import scope
import xgboost as xgb

df = pd.read_csv("./finaldf.csv")

selected_features = ['team_count','days_in_current_status','issue_type' ,\
                        'count_month_of_year','transictions_so_far','count_year']

# features are all columns without price_range
# note that there is no id column in this dataset
# here we have training features
X = df[selected_features].values
# and the targets
y = df.logsec_to_sol.values

def optimize(params, x, y):
    """
    The main optimization function.
    This function takes all the arguments from the search space
    and training features and targets. It then initializes
    the models by setting the chosen parameters and runs
    cross-validation and returns a negative accuracy score
    :param params: dict of params from hyperopt
    :param x: training data
    :param y: labels/targets
    :return: negative accuracy after 5 folds
    """
    # initialize model with current parameters
    model = ensemble.RandomForestRegressor(**params)
    # initialize stratified k-fold
    kf = model_selection.StratifiedKFold(n_splits=5)

    accuracies = []
 # loop over all folds
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]
        xtest = x[test_idx]
        ytest = y[test_idx]
        # fit model for current fold
        model.fit(xtrain, ytrain)
        #create predictions
        preds = model.predict(xtest)
        # calculate and append accuracy
        fold_accuracy = metrics.mean_squared_error(
            ytest,
            preds
        )
        accuracies.append(fold_accuracy)

    # return negative accuracy
    return -1 * np.mean(accuracies)


def rf_mse_cv(params, X=X, y=y,cv=5):
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators': int(params['n_estimators']), 
              'max_depth': int(params['max_depth']), 
             'max_features': params['max_features'],
             'eta': float(params['eta']),
             'gamma': float(params['gamma']),
             'booster': str(params['booster']),
             'min_child_weight': int(params['min_child_weight']),
             'subsample': float(params['subsample']),
             'colsample_bytree': float(params['colsample_bytree']),
             'lambda':float(params['lambda'])
             }
    
    # we use this params to create a new LGBM Regressor
    model = xgb.XGBRegressor(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).mean()

    return score

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("./finaldf.csv")

    selected_features = ['team_count','days_in_current_status','issue_type' ,\
                            'count_month_of_year','transictions_so_far','count_year']

    # features are all columns without price_range
    # note that there is no id column in this dataset
    # here we have training features
    X = df[selected_features].values
    # and the targets
    y = df.logsec_to_sol.values
    # define a parameter space
    # now we use hyperopt
    param_space = {
        # quniform gives round(uniform(low, high) / q) * q
        # we want int values for depth and estimators

        "max_depth": scope.int(hp.quniform("max_depth", 1, 15, 1)),
        "n_estimators": scope.int(
        hp.quniform("n_estimators", 100, 1500, 1)
        ),
        # choice chooses from a list of values
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        # uniform chooses a value between two values
        "max_features": hp.uniform("max_features", 0, 1),
        'eta': hp.quniform('eta', 0.01, 0.1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'booster': 'gbtree',
        'min_child_weight': hp.quniform('min_child_weight', 1, 7, 3),
        'subsample': hp.quniform('subsample', 0.6, 1.0, 0.1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 1.0, 0.1),
        'lambda': hp.quniform('lambda', 0.1, 1.0, 0.1),
    }
    # partial function
    optimization_function = partial(
        optimize,
        x=X,
        y=y
    )
    # initialize trials to keep logging information
    trials = Trials()

    # run hyperopt
    n_iter = 50
    random_state = 42
    hopt=fmin(fn=rf_mse_cv, # function to optimize
          space=param_space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=n_iter, # maximum number of iterations
          trials=trials, # logging
          rstate=np.random.RandomState(random_state) # fixing random state for the reproducibility
    )

    # model = ensemble.RandomForestRegressor(n_estimators=int(hopt['n_estimators']),
    #                     max_depth=int(hopt['max_depth']),max_features=int(hopt['max_features']),criterion=str(hopt['max_features']))

    # model.fit(X,y)
    # tpe_test_score=mean_squared_error(test_targets, model.predict(test_data))


    print("Best MSE {:.3f} params {}".format( rf_mse_cv(hopt,X,y), hopt))