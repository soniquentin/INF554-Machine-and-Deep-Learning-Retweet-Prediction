from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb

from tools import *
from os.path import exists
import os
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #To ignore furturewarning


def process_model(model_name = "rf2", compared_model = "", disagree_window = 10, debug = True) :
    """
        Process an existing model for sumbission :
            - Import the existing model
            - Create the csv file
            - Calculate the distance with the previous submission (Mean absolute error)

        INPUT :
            model_name : the pickle file name of the existing model
            compared_model : name of CSV file of a previous submission to which we will compare the new model
            disagree_window : show the disagree_window biggest absolute differences of prediction with the previous submissions
            debug : to print

        UTILIZATION :
            process_model(model_name = "rf2")
    """
    #Import the existing model and all the training objects
    rf, training_objs = import_model(model_name = model_name, debug = debug)

    #Import evaluation data and calculate the features
    X, embed, y, tweed_id = import_features_data(data = "data/evaluation.csv",
                                                        model_name = model_name,
                                                        training_objs = training_objs,
                                                        debug = debug) #y = None and training_objs = None

    #Make the prediction and write the submission file (+ compare it to a previous submission)
    write_and_compare_prediction(rf, X, embed, tweed_id, model_name, compared_model, disagree_window, debug)


def random_search(alg = "RF", n_iter = 100, cv = 3, save_model = True, model_name = "rf2", debug = True) :
    """
        Process a random search process and return (+ save) the model

        INPUT :
            n_iter : Number of combinations tested
            cv :  number of folds to use for cross validation (more cv folds reduces the chances of overfitting)
            save_model : True if the final estimator is saved (.pickle)
            model_name : model name to be saved (.pickle)
            debug : True if want to print for debug

        OUTPUT :
            Best estimator rf

        UTILIZATION :
            rf = random_search()
    """

    X,embed,y,tweet_id = import_features_data(debug = debug)


    #Param grid
    if alg == "RF" :
        param_grid = {
        'n_estimators': [100*i for i in range(1,11)], # the number of trees in the forest
        'max_features': ['log2', 'sqrt', 1.0], # number of features to consider at every split
        'max_depth' : [10*i for i in range(1,11)], # maximum number of levels in tree
        'min_samples_split' : [2, 5, 10], # the minimum number of samples required to split an internal node
        'min_samples_leaf' : [1, 2, 4], # the minimum number of samples required to be at a leaf node
        'bootstrap' : [True, False], # Whether bootstrap samples are used when building trees
        'random_state' : [41]
        }
        rf = RandomForestRegressor()
    elif alg == "GB" :
        rf = GradientBoostingRegressor()
    else :
        param_grid = {'booster':['gbtree','gblinear'],
                  'learning_rate': [0.1, 0.3],
                  'max_depth': [10*i for i in range(1,11)],
                  'min_child_weight': [10,15,20,25],
                  'n_estimators': [300,400,500,600],
                  "reg_alpha"   : [0.5,0.2,1],
                  "reg_lambda"  : [2,3,5],
                  "gamma"       : [1,2,3]}
        rf = xgb.XGBRegressor()

    rs = RandomizedSearchCV(estimator = rf,
                param_distributions = param_grid,
                n_iter = n_iter, #Number of combinations tested
                cv = cv, #number of folds to use for cross validation (more cv folds reduces the chances of overfitting)
                verbose = 2, #Quantity of msg print
                random_state = 42, #Pseudo random number generator state used for random uniform sampling
                n_jobs = -1, #Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
                scoring = 'neg_mean_absolute_error')

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
            print("\n==== SAVING MODEL ====")
            print("    --> best parameters :", rs.best_params_)
        with open(os.path.dirname(__file__) + "/models/" + model_name, 'wb') as f :
            pickle.dump(rs.best_estimator_, f)


    return rs.best_estimator_ #Return the best estimator rf_best




if __name__ == "__main__":
    ##==========================================##

    model_name = "rf8"

    """
    #### ========  RF  ========
    scaler = train_model(alg = "RF",
                        model_name = model_name ,
                        #loss = "absolute_error",  #GB
                        bootstrap = True,  #RF
                        max_depth = 90,
                        max_features = 1.0,
                        min_samples_leaf = 2,
                        min_samples_split = 2,
                        n_estimators = 300,
                        #objective = "reg:absoluteerror", #XGB
                        #eval_metric = "mae" #XGB
                        random_state = 42)
    """

    #### ========  NNnetwork  ========
    """
    train_model(alg = "NN",
                model_name = model_name ,
                epochs=100,
                batch_size=32,
                validation_split = 0.2)

    process_model(model_name = model_name, compared_model = "third_submission", disagree_window = 15)
    process_model(model_name = model_name, compared_model = "top_score", disagree_window = 15)
    """

    ##==========================================##
    random_search(n_iter = 75, model_name = model_name)
