from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from tools import *
from os.path import exists
import os
import pickle

import matplotlib.pyplot as plt


if __name__ == "__main__":
    n_estimators_list = [100 + 200*i for i in range(20)]

    filename = "evaluation_numberized"
    file_exists = exists(os.path.dirname(__file__) + "/output/" + filename)
    if file_exists :
        f = open(os.path.dirname(__file__) + "/output/" + filename, 'rb')
        data_numberized = pickle.load(f)
        f.close()
    else :
        data_numberized = pd.read_csv(data)
        data_numberized = numberize_features(data_numberized)

        with open(os.path.dirname(__file__) + "/output/" + filename, 'wb') as f :
            pickle.dump(data_numberized, f)

    X = data_numberized.drop(['retweets_count'], axis = 1, inplace = False )
    y = data_numberized["retweets_count"]

    t_timeline = []
    y_timeline = []

    for n_estimator in n_estimators_list :
        print("n_estimators :", n_estimator)
        sum = 0
        for i in range(1,3):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=21*i)
            rf = RandomForestRegressor(n_estimators = n_estimator, random_state = 21*i)

            rf.fit(X_train, y_train)
            sum += evaluation(rf, X_test, y_test)

        t_timeline.append(sum/3)
        y_timeline.append(n_estimator)

    plt.plot(t_timeline, y_timeline)
    plt.show()
