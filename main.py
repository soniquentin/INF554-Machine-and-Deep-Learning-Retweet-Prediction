from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from tools import *
from os.path import exists
import os
import pickle


def simple_train_example(X,y) :
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
    rf.fit(X_train, y_train)
    evaluation(rf, X_test, y_test)


def random_search(n_iter = 100, cv = 3, filename = "evaluation_numberized", data = "data/train.csv", save_model = True, model_name = "rf", debug = True) :
    """
        Process a random search process and return (+ save) the model

        INPUT :
            n_iter : Number of combinations tested
            cv : number of folds to use for cross validation (more cv folds reduces the chances of overfitting)
            filename : file name (.pickle) of the numberized data
            data : file name of the raw data (.csv)
            debug : True if want to print for debug

        OUTPUT :
            Best estimator rf

        UTILIZATION :
            rf = random_search()
    """

    if debug :
        print("\n==== CREATING NUMBERIZED DATA ====")

    file_exists = exists(os.path.dirname(__file__) + "/output/" + filename)
    if file_exists :
        if debug :
            print("    --> numberized data already exist (Path : {})".format(os.path.dirname(__file__) + "/output/" + filename))

        f = open(os.path.dirname(__file__) + "/output/" + filename, 'rb')
        data_numberized = pickle.load(f)
        f.close()
    else :
        if debug :
            print("    --> numberizing ... and saving at path {}...".format(os.path.dirname(__file__) + "/output/" + filename))

        data_numberized = pd.read_csv(data)
        data_numberized = numberize_features(data_numberized)

        with open(os.path.dirname(__file__) + "/output/" + filename, 'wb') as f :
            pickle.dump(data_numberized, f)

    X = data_numberized.drop(['retweets_count'], axis = 1, inplace = False )
    y = data_numberized["retweets_count"]

    #Param grid
    param_grid = {
    'n_estimators': [100*i for i in range(1,21)], # the number of trees in the forest
    'max_features': ['log2', 'sqrt', 1.0], # number of features to consider at every split
    'max_depth' : [10*i for i in range(1,11)], # maximum number of levels in tree
    'min_samples_split' : [2, 5, 10], # the minimum number of samples required to split an internal node
    'min_samples_leaf' : [1, 2, 4], # the minimum number of samples required to be at a leaf node
    'bootstrap' : [True, False] # Whether bootstrap samples are used when building trees
    }


    rf = RandomForestRegressor()

    rs = RandomizedSearchCV(estimator = rf,
                param_distributions = param_grid,
                n_iter = n_iter, #Number of combinations tested
                cv = cv, #number of folds to use for cross validation (more cv folds reduces the chances of overfitting)
                verbose = 2, #Quantity of msg print
                random_state = 42, #Pseudo random number generator state used for random uniform sampling
                n_jobs = -1) #Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

    if debug :
        print("\n==== RANDOM SEARCH ====")
        print("    --> param grid :", param_grid)
        print("    --> searching...")

    rs.fit(X, y)

    if debug :
        print("\n==== RESULT ====")
        print("    --> best parameters :", rs.best_params_)

    if save_model :
        if debug :
            print("\n==== SAVE MODEL ====")
            print("    --> best parameters :", rs.best_params_)
        with open(os.path.dirname(__file__) + "/output/" + model_name, 'wb') as f :
            pickle.dump(rs.best_estimator_, f)


    return rs.best_estimator_ #Return the best estimator rf_best


def write_prediction(rf, X, filename = "rf_pred.txt") :
    """
        Create the file for submission

        INPUT :
            rf : estimator
            X : Features of retweet to predict
            file : the text file name that will be created

        UTILIZATION :
    """
    predictions = rf.predict(X)

    X["predictions"] = predictions

    with open('data/' + filename, 'w') as f:
        f.write('TweetID,retweets_count\n')
        for index, row in X.iterrows():
            f.write('{},{}\n'.format(row["TweetID"], int(row["predictions"]) ))



if __name__ == "__main__":
    random_search()
