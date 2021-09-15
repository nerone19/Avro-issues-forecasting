import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
import numpy as np
'''Initialize all the regesssion models object we are interested in'''
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler,Normalizer
import matplotlib.pyplot as plt
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor


class GreedyFeatureSelection: 
        """
        A simple and custom class for greedy feature selection.
        You will need to modify it quite a bit to make it suitable
        for your dataset.
        """
        def evaluate_score(self, X, y):
            """
            This function evaluates model on data and returns
            Area Under ROC Curve (AUC)
            NOTE: We fit the data and calculate AUC on same data.
            WE ARE OVERFITTING HERE. 
            But this is also a way to achieve greedy selection.
            k-fold will take k times longer.
            If you want to implement it in really correct way,you
            calculate OOF AUC and return mean AUC over k folds.
            This requires only a few lines of change and has been 
            shown a few times in this book.
            :param X: training data
            :param y: targets
            :return: overfitted area under the roc curve
            """
            # fit the logistic regression model,
            # and calculate AUC on same data
            # again: BEWARE
            # you can choose any model that suits your data

            model = LinearRegression()
            model.fit(X, y)
            scores = cross_val_score(model,X=X, y=y, cv=10, n_jobs=1,scoring='neg_mean_squared_error')
            return abs(np.mean(scores))

        def _feature_selection(self, X, y):
            """
            This function does the actual greedy selection
            :param X: data, numpy array
            :param y: targets, numpy array
            :return: (best scores, best features)
            """
            # initialize good features list 
            # and best scores to keep track of both
            good_features = []
            best_scores = []

            # calculate the number of features
            num_features = X.shape[1]

            # infinite loop
            while True:
                # initialize best feature and score of this loop
                this_feature = None
                best_score = np.inf
                # loop over all features
                for feature in range(num_features):
                # if feature is already in good features,
                    # skip this for loop
                    if feature in good_features:
                            continue
                    # selected features are all good features till now
                    # and current feature
                    selected_features = good_features + [feature]
                    # remove all other features from data
                    xtrain = X[:, selected_features]
                    # calculate the score, in our case, AUC
                    score = self.evaluate_score(xtrain, y)
                    print(score)
                    # if score is greater than the best score
                    # of this loop, change best score and best feature
                    if score < best_score:
                        this_feature = feature
                        best_score = score
                    # if we have selected a feature, add it
                    
                                    # to the good feature list and update best scores list
                    if this_feature != None:
                        good_features.append(this_feature)
                        best_scores.append(best_score)
                    # if we didnt improve during the last two rounds,
                    # exit the while loop
                    if len(best_scores) > 25:
                        if best_scores[-1] < best_scores[-2]:
                            break
                # return best scores and good features
                # why do we remove the last data point?
                return best_scores[:-1], good_features[:-1]
        
        def __call__(self, X, y):
            """
            Call function will call the class on a set of arguments
            """
            # select features, return scores and selected indices
            scores, features = self._feature_selection(X, y)
            # transform data with selected features
            return X[:, features], scores, features



# '''set a seed for reproducibility'''
# seed = 44


# ''''We are interested in the following 14 regression models.
# All initialized with default parameters except random_state and n_jobs.'''
# lr = LinearRegression(n_jobs = -1)
# lasso = Lasso(random_state = seed)
# ridge = Ridge(random_state = seed)
# elnt = ElasticNet(random_state = seed)
# kr = KernelRidge()
# dt = DecisionTreeRegressor(random_state = seed)
# svr = SVR()
# knn = KNeighborsRegressor(n_jobs= -1)
# pls = PLSRegression()
# rf = RandomForestRegressor(n_jobs = -1, random_state = seed)
# et = ExtraTreesRegressor(n_jobs = -1, random_state = seed)
# ab = AdaBoostRegressor(random_state = seed)
# gb = GradientBoostingRegressor(random_state = seed)
# xgb = XGBRegressor(n_jobs = -1, random_state = seed)
# lgb = LGBMRegressor(n_jobs = -1, random_state = seed)


if __name__ == "__main__":
    df = pd.read_csv (r'./finaldf.csv')
    X_train=df.drop(['year','logsec_to_sol','key','project','sec_to_sol','when',\
                       'updated','calendar_day','transition','who','created','reporter','updated','resolution',\
                       'resolutiondate'],axis=1)
    y_train=df[['logsec_to_sol']]

    scaler = StandardScaler().fit(X_train, y_train)

    X_train_st= scaler.transform(X_train)

    col_names = X_train.columns

    # transform data by greedy feature selection
    X_transformed, scores,which_features = GreedyFeatureSelection()(X_train.to_numpy() ,y_train)

    plt.plot(range(len(scores)), scores)
    plt.show()
    print(scores)

    print([X_train.columns[f] for f in list(set(which_features))])
    print("{} feature were kept out of {}".format( len(list(set(which_features))),len(X_train.columns)))


    from sklearn.feature_selection import RFE
    model = LinearRegression()
    rfe = RFE(
            estimator=model,
            n_features_to_select=3
    )
    # fit RFE
    rfe.fit(X_train_st, y_train)
    # get the transformed data with
    # selected columns
    X_transformed = rfe.transform(X_train_st)
    scores = cross_val_score(model,X=X_transformed, y=y_train, cv=10, n_jobs=1,scoring='neg_mean_squared_error')
    print(np.mean(scores))