from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
import matplotlib.pyplot as plt

from tools import *
from os.path import exists
import os
import pickle

#### PARAMETERS ####
compared_model = "third_submission"

####################

def simple_train_example() :

    data_numberized = import_features_data(debug = True)
    X = data_numberized.drop(['retweets_count'], axis = 1, inplace = False )
    y = data_numberized["retweets_count"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    rf = RandomForestRegressor(n_estimators = 100)
    rf.fit(X_train, y_train)
    evaluation(rf, X_test, y_test)



def process_model(model_name = "rf2", compared_model = compared_model, debug = True) :
    """
        Process an existing model for sumbission :
            - Import the existing model
            - Create the csv file
            - Calculate the distance with the previous submission (Mean square error)

        INPUT :
            model_name : the pickle file name of the existing model
            compared_model : name of CSV file of a previous submission to which we will compare the new model
            debug : to print

        UTILIZATION :
            process_model(model_name = "rf2")
    """

    X = import_features_data(data = "data/evaluation.csv", debug = debug)
    rf = import_model(model_name = model_name, debug = debug)
    write_and_compare_prediction(rf, X, model_name, compared_model, 10, debug)


def search_minimum_hp(hp, range_hp, plot = True, cv = 3, nb_treads = 5, debug = True):
    from threading import Thread, Lock

    n_iter = len(range_hp)
    score_mean_list = [0]*n_iter

    data_numberized = import_data(debug = debug)

    X = data_numberized.drop(['retweets_count'], axis = 1, inplace = False )
    y = data_numberized["retweets_count"]


    def thread_run(lock, index_threads, nb_treads) :
        for i in range(n_iter) :
            if i%nb_treads == index_threads :
                if debug :
                    print("    --> [Thread {}] Computing {}".format(index_threads,i))
                rf = RandomForestRegressor(**{hp : range_hp[i], "max_depth" : 100})
                scores = cross_val_score(rf, X, y, cv = cv, scoring = 'neg_mean_absolute_error')
                with lock :
                    score_mean_list[i] = scores.mean()

    lock = Lock()
    threads = [Thread(target=thread_run, args=(lock, i,nb_treads,)) for i in range(nb_treads)]

    for thread in threads :
        thread.start()

    for thread in threads :
        thread.join()

    if plot :
        plt.plot(range_hp, score_mean_list)
        plt.savefig('output/search_{}.png'.format(hp), dpi = 300)
        plt.show()



def random_search(n_iter = 100, cv = 3, save_model = True, model_name = "rf2", debug = True) :
    """
        Process a random search process and return (+ save) the model

        INPUT :
            n_iter : Number of combinations tested
            cv :  number of folds to use for cross validation (more cv folds reduces the chances of overfitting)
            save_model : True if the final estimator is saved (.pickle)
            model_name : model name (.pickle)
            debug : True if want to print for debug

        OUTPUT :
            Best estimator rf

        UTILIZATION :
            rf = random_search()
    """

    data_numberized = import_data(debug = debug)

    X = data_numberized.drop(['retweets_count'], axis = 1, inplace = False )
    y = data_numberized["retweets_count"]

    #Param grid
    param_grid = {
    'n_estimators': [100*i for i in range(1,11)], # the number of trees in the forest
    'max_features': ['log2', 'sqrt', 1.0], # number of features to consider at every split
    'max_depth' : [10*i for i in range(1,11)], # maximum number of levels in tree
    'min_samples_split' : [2, 5, 10], # the minimum number of samples required to split an internal node
    'min_samples_leaf' : [1, 2, 4], # the minimum number of samples required to be at a leaf node
    'bootstrap' : [True, False], # Whether bootstrap samples are used when building trees
    'random_state' : [42],
    'criterion' : ["absolute_error"]
    }


    rf = RandomForestRegressor()

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
            print("\n==== SAVE MODEL ====")
            print("    --> best parameters :", rs.best_params_)
        with open(os.path.dirname(__file__) + "/models/" + model_name, 'wb') as f :
            pickle.dump(rs.best_estimator_, f)


    return rs.best_estimator_ #Return the best estimator rf_best



if __name__ == "__main__":

    #random_search(n_iter = 75, model_name = "rf3")

    #search_minimum_hp(hp = 'n_estimators', range_hp = [100*i for i in range(1,21)], plot = True, cv = 3, nb_treads = 5, debug = True)
    process_model(model_name = "rf")
