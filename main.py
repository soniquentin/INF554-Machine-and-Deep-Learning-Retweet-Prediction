from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
import matplotlib.pyplot as plt

from tools import *
from os.path import exists
import os
import pickle


def simple_train_example() :

    data_numberized = import_data(debug = True)
    X = data_numberized.drop(['retweets_count'], axis = 1, inplace = False )
    y = data_numberized["retweets_count"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    rf = RandomForestRegressor(n_estimators = 100)
    rf.fit(X_train, y_train)
    evaluation(rf, X_test, y_test)


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
        plt.show()



def random_search(n_iter = 100, cv = 3, save_model = True, model_name = "rf", debug = True) :
    """
        Process a random search process and return (+ save) the model

        INPUT :
            n_iter : Number of combinations tested
            cv :  number of folds to use for cross validation (more cv folds reduces the chances of overfitting)
            save_model : True if the final estimator is saved (.pickle)

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
    'random_state' : [42]
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

    #search_minimum_hp(hp = 'n_estimators', range_hp = [100*i for i in range(1,21)], plot = True, cv = 3, nb_treads = 5, debug = True)

    random_search(n_iter = 75)

    #search_minimum_hp(hp = 'n_estimators', range_hp = [100*i for i in range(1,21)], plot = True, cv = 3, nb_treads = 5, debug = True)
    #simple_train_example()
