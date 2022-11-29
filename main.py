from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from tools import *
from os.path import exists
import pickle



def random_search(n_iter = 100, cv = 3, filename = "evaluation_numberized", data = "data/train.csv") :
    file_exists = exists(filename)
    if file_exists :
        f = open(filename, 'rb')
        data_numberized = pickle.load(f)
        f.close()
    else :
        data_numberized = pd.read_csv(data)
        data_numberized = numberize_features(data_numberized)

        with open(filename, 'wb') as f :
            pickle.dump(data_numberized, f)

    X = data_numberized.drop(['retweets_count'], axis = 1, inplace = False )
    y = data_numberized["retweets_count"]

    #Param grid
    param_grid = {
    'n_estimators': [100*i for i in range(1,21)], # the number of trees in the forest
    'max_features': ['auto', 'sqrt'], # number of features to consider at every split
    'max_depth' : [10*i for i in range(1,11)], # maximum number of levels in tree
    'min_samples_split' : [2, 5, 10], # the minimum number of samples required to split an internal node
    'min_samples_leaf' : [1, 2, 4], # the minimum number of samples required to be at a leaf node
    'bootstrap' : [True, False] # Whether bootstrap samples are used when building trees
    }

    rf = RandomForestClassifier()

    rs = RandomizedSearchCV(estimator = rf,
                param_distributions = param_grid,
                n_iter = 100, #Number of combinations tested
                cv = 3, #number of folds to use for cross validation (more cv folds reduces the chances of overfitting)
                verbose = 2, #Quantity of msg print
                random_state = 42, #Pseudo random number generator state used for random uniform sampling
                n_jobs = -1) #Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.


    print("Best parameters : ", rs.best_params_)

    return rs.best_estimator_ #Return the best estimator rf_best


"""
X_test = numberize_features(X_test)

print("TRAIN")
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)

print("PREDICT")
predictions = rf.predict(X_test)

X_test["predictions"] = predictions

with open('data/rf3.txt', 'w') as f:
    f.write('TweetID,retweets_count\n')
    for index, row in X_test.iterrows():
        f.write('{},{}\n'.format(row["TweetID"], int(row["predictions"]) ))
"""

"""
print("TRAIN")

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X_train2, y_train2)
evaluation(rf, X_test2, y_test2)
"""


if __name__ == "__main__":
    random_search()
